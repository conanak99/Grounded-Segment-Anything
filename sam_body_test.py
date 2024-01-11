import matplotlib.pyplot as plt
import warnings
from segment_anything import build_sam, build_sam_hq, SamPredictor
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.models import build_model
from PIL import Image
from huggingface_hub import hf_hub_download
from EfficientSAM.LightHQSAM.setup_light_hqsam import setup_model

import cv2
import os
import sys
import time
import torch
import numpy as np

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
        (mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


print("Loading GroundingDino model from HuggingFace Hub...")
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(
    ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)
print("GroundingDino Model loaded.")

print("Loading SAM Predictor...")
hq_sam_predictor = SamPredictor(
    build_sam_hq(checkpoint="sam_hq_vit_h.pth").to(device))
print("SAM Models loaded.")

sample_folder = 'samples'
file_names = os.listdir(sample_folder)
test_folder = 'test_body'
# get files only, ignore folders
file_names = [file_name for file_name in file_names if '.' in file_name]

for file_name in file_names:
    try:
        print(f'Testing {file_name}...')

        local_image_path = f'{sample_folder}/{file_name}'
        result_file_path = f'samples/{test_folder}/{file_name}'

        # get file name without extension

        # continue loop if result file already exists
        if os.path.exists(result_file_path):
            print(f'{file_name} result existed.')
            continue

        file_name_no_extension = os.path.splitext(file_name)[0]
        body_mask_file_path = f'samples/{test_folder}/{file_name_no_extension}_body_mask.png'
        head_mask_file_path = f'samples/{test_folder}/{file_name_no_extension}_head_mask.png'
        final_mask_file_path = f'samples/{test_folder}/{file_name_no_extension}_final_mask.png'

        image_source, image = load_image(local_image_path)

        # Plot original images, mask and output images
        total_rows = 4
        fig = plt.figure()
        plt.subplot(1, total_rows, 1)
        plt.imshow(Image.open(local_image_path))
        plt.title("Original Image")
        plt.axis("off")

        # Detect sections with DINO
        start_time = time.time()
        print("Detecting body...")
        annotated_frame, detected_boxes = detect(
            image, text_prompt="body", model=groundingdino_model)

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time [detect body]: {:.2f} seconds".format(
            execution_time))

        # Segment with Dino input + SAM

        start_time = time.time()
        print("Segmenting body...")
        segmented_frame_masks = segment(
            image_source, hq_sam_predictor, boxes=detected_boxes)

        body_mask = segmented_frame_masks[0][0].cpu().numpy()

        plt.subplot(1, total_rows, 2)
        plt.imshow(Image.fromarray(body_mask))
        Image.fromarray(body_mask).save(body_mask_file_path)
        plt.axis("off")

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time [segment body]: {:.2f} seconds".format(
            execution_time))

        start_time = time.time()
        print("Detecting body...")
        annotated_frame, detected_boxes = detect(
            image, text_prompt="head,face", model=groundingdino_model)

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time [face and hair]: {:.2f} seconds".format(
            execution_time))

        # Segment with Dino input + SAM

        start_time = time.time()
        print("Segmenting face and hair...")
        segmented_frame_masks = segment(
            image_source, hq_sam_predictor, boxes=detected_boxes)

        head_mask = segmented_frame_masks[0][0].cpu().numpy()

        plt.subplot(1, total_rows, 3)
        plt.imshow(Image.fromarray(head_mask))
        plt.axis("off")
        Image.fromarray(head_mask).save(head_mask_file_path)

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time [segment face and hair]: {:.2f} seconds".format(
            execution_time))

        # Remove face and hair mash from body mask
        body_mask = cv2.imread(body_mask_file_path, cv2.IMREAD_GRAYSCALE)
        head_mask = cv2.imread(head_mask_file_path, cv2.IMREAD_GRAYSCALE)
        final_mask = body_mask - np.bitwise_and(body_mask, head_mask)
        Image.fromarray(final_mask).save(final_mask_file_path)

        annotated_frame_with_mask = draw_mask(
            final_mask, annotated_frame)
        plt.subplot(1, total_rows, 4)
        plt.imshow(Image.fromarray(annotated_frame_with_mask))
        plt.axis("off")

        fig.tight_layout()
        plt.savefig(result_file_path, dpi=300)
    except Exception as e:
        print(f'Error: {e}')
    # plt.get_current_fig_manager().full_screen_toggle()
    # plt.show(block=False)
