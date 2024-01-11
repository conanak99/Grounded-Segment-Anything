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

load_dotenv()
warnings.filterwarnings("ignore")

local_image_path = "download/1704866704_0.jpg"
mask_expand_image_path = 'download/1704866704_2_mask_expand.png'

# Convert image to base64
with open(local_image_path, "rb") as image_file:
    encoded_image = "data:image/jpeg;base64," + \
        base64.b64encode(image_file.read()).decode('utf-8')
with Image.open(local_image_path) as im:
    width, height = im.size
with open(mask_expand_image_path, "rb") as image_file:
    encoded_mask = "data:image/png;base64," + \
        base64.b64encode(image_file.read()).decode('utf-8')

random_id = str(time.time())[:10]

# Inpaint
start_time = time.time()
print("Inpainting...")
client = NovitaClient(os.getenv("NOVITA_AI_API_KEY"))

generation_count = 2
req = Img2ImgRequest(
    init_images=[encoded_image],
    mask=encoded_mask,
    model_name='realisticVisionV40_v40VAE-inpainting_81543.safetensors',
    prompt='raw photo of a nude woman, (naked)',
    # model_name='meinahentai_v4-inpainting_86046.safetensors',
    # prompt='art of a nude woman, (naked)',
    negative_prompt='((clothing)), (monochrome:1.3), (deformed, distorted, disfigured:1.3), ((hair)), jeans, tattoo, wet, water, clothing, shadow, 3d render, cartoon, ((blurry)), duplicate, ((duplicate body parts)), (disfigured), (poorly drawn), ((missing limbs)), logo, signature, text, words, low res, boring, artifacts, bad art, gross, ugly, poor quality, low quality, poorly drawn, bad anatomy, wrong anatomy',
    sampler_name="DPM++ SDE Karras",
    cfg_scale=7,
    steps=20,
    width=width,
    height=height,
    seed=-1,
    denoising_strength=0.9,
    n_iter=generation_count,
    inpainting_fill=0,
    controlnet_units=[{
        "model": "control_v11p_sd15_openpose",
        "weight": 1,
        "input_image": encoded_image,
        "module": "openpose_full",
        "control_mode": 2
    }]
)
data = client.sync_txt2img(req).data
for i in range(generation_count):
    output_image_path = f'download/inpaint_test/{random_id}_3_output_{i}.png'
    save_image(data.imgs_bytes[i], output_image_path)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time [inpaint]: {:.2f} seconds".format(execution_time))

# Plot original images, mask and output images
total_rows = generation_count + 2
fig = plt.figure()
plt.subplot(1, total_rows, 1)
plt.imshow(Image.open(local_image_path))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, total_rows, 2)
plt.imshow(Image.open(mask_expand_image_path))
plt.title("Mask")
plt.axis("off")

# Show output images
for i in range(generation_count):
    output_image_path = f'download/inpaint_test/{random_id}_3_output_{i}.png'
    output_image = Image.open(output_image_path)
    plt.subplot(1, total_rows, i + 1 + 2)
    plt.imshow(output_image)
    plt.title(f"Output Image {i+1}")
    plt.axis("off")

fig.tight_layout()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.show()
