import warnings
from segment_anything import build_sam, SamPredictor
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

import os
import sys
import time
import base64
import torch
import requests
import numpy as np

from dotenv import load_dotenv
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


def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(BytesIO(r.content)) as im:
        # Resize image to maximum width of 600 pixels
        width, height = im.size
        if width > 600:
            new_width = 600
            new_height = int((new_width / width) * height)
            im = im.resize((new_width, new_height))

        im.save(image_file_path)
    print('Image downloaded from url: {} and saved to: {}.'.format(
        url, image_file_path))


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


ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
print("Loading GroundingDino model from HuggingFace Hub...")
groundingdino_model = load_model_hf(
    ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)
print("GroundingDino Model loaded.")

sam_checkpoint = 'sam_vit_h_4b8939.pth'
print("Loading SAM Predictor...")
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
print("SAM Model loaded.")

# Load image
local_image_path = "download/inpaint_demo.jpg"
mask_image_path = "download/mask.png"
clothes_prompts = "clothes, clothing"  # or bra, panties, clothes

# image_url = "https://pbs.twimg.com/media/GCenbwybcAErMs5?format=jpg&name=900x900"
while True:
    user_input = input("Enter image URL (or 'quit' to exit): ")
    if user_input == "quit":
        print("Quit program")
        break

    image_url = user_input
    download_image(image_url, local_image_path)
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
    start_time = time.time()
    print("Segmenting...")
    segmented_frame_masks = segment(
        image_source, sam_predictor, boxes=detected_boxes)
    annotated_frame_with_mask = draw_mask(
        segmented_frame_masks[0][0], annotated_frame)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time [segment]: {:.2f} seconds".format(execution_time))

    mask = segmented_frame_masks[0][0].cpu().numpy()
    image_mask_pil = Image.fromarray(mask)
    mask_image_path = "download/mask.png"
    image_mask_pil.save(mask_image_path)

    # Convert image to base64
    with open(local_image_path, "rb") as image_file:
        encoded_image = "data:image/jpeg;base64," + \
            base64.b64encode(image_file.read()).decode('utf-8')
    with Image.open(local_image_path) as im:
        width, height = im.size
    with open(mask_image_path, "rb") as image_file:
        encoded_mask = "data:image/png;base64," + \
            base64.b64encode(image_file.read()).decode('utf-8')

    # Inpaint
    start_time = time.time()
    print("Inpainting...")
    client = NovitaClient(os.getenv("NOVITA_AI_API_KEY"))
    output_image_path = "download/output.png"
    req = Img2ImgRequest(
        model_name='realisticVisionV40_v40VAE-inpainting_81543.safetensors',
        init_images=[encoded_image],
        mask=encoded_mask,
        prompt='raw photo of a nude woman, (naked)',
        negative_prompt='((clothing)), (monochrome:1.3), (deformed, distorted, disfigured:1.3), (hair), jeans, tattoo, wet, water, clothing, shadow, 3d render, cartoon, ((blurry)), duplicate, ((duplicate body parts)), (disfigured), (poorly drawn), ((missing limbs)), logo, signature, text, words, low res, boring, artifacts, bad art, gross, ugly, poor quality, low quality, poorly drawn, bad anatomy, wrong anatomy​',
        sampler_name="DPM++ SDE Karras",
        cfg_scale=7,
        steps=20,
        width=width,
        height=height,
        seed=-1,
        denoising_strength=0.,
        inpainting_fill=0
    )
    save_image(client.sync_txt2img(req).data.imgs_bytes[0], output_image_path)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time [input]: {:.2f} seconds".format(execution_time))

    output_image = Image.open(output_image_path)
    output_image.show()
