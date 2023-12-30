import matplotlib.pyplot as plt
import warnings
from segment_anything import build_sam, build_sam_hq, SamPredictor
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.models import build_model
import GroundingDINO.groundingdino.datasets.transforms as T
from PIL import Image
from novita_client import NovitaClient, Img2ImgRequest, save_image
from huggingface_hub import hf_hub_download
from io import BytesIO
from EfficientSAM.LightHQSAM.setup_light_hqsam import setup_model

import os
import cv2
import sys
import time
import base64
import torch
import requests
import numpy as np

from dotenv import load_dotenv
from PIL import Image, ImageFilter
import os

load_dotenv()
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(
        repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(
        checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def detect(image, text_prompt, model, box_threshold=0.3, text_threshold=0.25):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )

    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
    return annotated_frame, boxes


def segment(image, sam_model, boxes):
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_model.transform.apply_boxes_torch(
        boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks.cpu()


def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray(
        (mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


print("Loading GroundingDino model from HuggingFace Hub...")
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(
    ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)
print("GroundingDino Model loaded.")

print("Loading SAM Predictor...")
sam_predictor = SamPredictor(
    build_sam(checkpoint='sam_vit_h_4b8939.pth').to(device))

# #Sam HQ, 5-10% slower only
# https://github.com/SysCV/sam-hq?tab=readme-ov-file#model-checkpoints
# Might need to change the code of build_sam_hq to load from cpu lol
hq_sam_predictor = SamPredictor(
    build_sam_hq(checkpoint="sam_hq_vit_h.pth").to(device))

checkpoint = torch.load('sam_hq_vit_tiny.pth',
                        map_location=torch.device(device))
light_hqsam = setup_model()
light_hqsam.load_state_dict(checkpoint, strict=True)
light_hqsam.to(device=device)
light_sam_predictor = SamPredictor(light_hqsam)

print("3 SAM Models loaded.")

# Load image
clothes_prompts = "clothes, clothing"  # or bra, panties, clothes

sample_folder = 'samples'
file_names = os.listdir(sample_folder)
file_names = [
    file_name for file_name in file_names if 'test_result' not in file_name]

for file_name in file_names:
    try:
        print(f'Testing {file_name}...')

        local_image_path = f'{sample_folder}/{file_name}'
        result_file_path = f'samples/test_result/{file_name}'

        # continue loop if result file already exists
        if os.path.exists(result_file_path):
            print(f'{file_name} result existed.')
            continue

        image_source, image = load_image(local_image_path)

        # Detect sections with DINO
        start_time = time.time()
        print("Detecting...")
        annotated_frame, detected_boxes = detect(
            image, text_prompt=clothes_prompts, model=groundingdino_model)

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time [detect]: {:.2f} seconds".format(execution_time))

        # Segment with Dino input + SAM
        sam_predictors = {
            "light_sam_predictor": light_sam_predictor,
            "sam_predictor": sam_predictor,
            "hq_sam_predictor": hq_sam_predictor
        }

        # Plot original images, mask and output images
        total_rows = 4
        fig = plt.figure()
        plt.subplot(1, total_rows, 1)
        plt.imshow(Image.open(local_image_path))
        plt.title("Original Image")
        plt.axis("off")

        i = 2
        for predictor_name, predictor in sam_predictors.items():
            start_time = time.time()
            print(f"Segmenting with {predictor_name}...")
            segmented_frame_masks = segment(
                image_source, predictor, boxes=detected_boxes)

            annotated_frame_with_mask = draw_mask(
                segmented_frame_masks[0][0], annotated_frame)

            output_image = Image.fromarray(annotated_frame_with_mask)

            end_time = time.time()
            execution_time = end_time - start_time
            print(
                f"Execution time [segment with {predictor_name}]: {execution_time:.2f} seconds.")
            plt.subplot(1, total_rows, i)
            plt.imshow(output_image)
            plt.title(predictor_name)
            plt.axis("off")
            i = i + 1

        fig.tight_layout()
        plt.savefig(result_file_path, dpi=300)
    except Exception as e:
        print(f'Error: {e}')
    # plt.get_current_fig_manager().full_screen_toggle()
    # plt.show(block=False)
