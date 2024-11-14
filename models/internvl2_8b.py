# transformers=4.37
# env: minigpt4v

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = '/data/xiangyu/benchmarkModels/CHENCHEN/huggingface/hub/models--OpenGVLab--InternVL2-8B/snapshots/c527cd3717f4bb8339002b342bc9476ec0485004'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=1024, do_sample=True)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

@app.route('/generate', methods=['POST'])
def generate_text():
    # Get prompt from the request
    data = request.json
    prompt = data.get("prompt", "")
    image_paths = data.get("image_paths", [])

    # no image
    if len(image_paths) == 0:
        print(f"""
            This is the Llama-3___2-90B-Vision-Instruct.py script.
            The prompt is: {prompt}
            The image path is: {image_paths}
        """)
        response, history = model.chat(tokenizer, None, prompt, generation_config,
                                history=None, return_history=True)
        print(response)

        return jsonify(response)
    else:
        images = []
        for i in range(len(image_paths)):
            images.append(load_image(image_paths[i], max_num=12).to(torch.bfloat16).cuda())
        pixel_values = torch.cat(images, dim=0)
        num_patches_list = []
        for i in range(len(image_paths)):
            num_patches_list.append(images[i].size(0))

        image_prompts = []
        for i in range(len(image_paths)):
            image_prompts.append(f"Image-{i+1}: <image>\n")
        image_prompt = "".join(image_prompts)
        prompt = image_prompt + " " + prompt

        print(f"""
            This is the Llama-3___2-90B-Vision-Instruct.py script.
            The prompt is: {prompt}
            The image path is: {image_paths}
        """)

        response, history = model.chat(tokenizer, pixel_values, prompt, generation_config,
                                        num_patches_list=num_patches_list,
                                        history=None, return_history=True)

        print(response)

        return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5042)