# modality: text
import sys
import torch
# update transformers using "pip install --upgrade transformers"
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download
from flask import Flask, request, jsonify
import os
from HPTModel.hpt1_5 import HPT1_5
# from zmq import device

app = Flask(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_grad_enabled(False)


class HPT:
    def __init__(self, model_id, device, fp16):
        torch_dtype = torch.float16 if fp16 else torch.float32
        if 'air' in model_id:
            self.model = HPT1_5(global_model_path=model_id,
                                prompt_template='llama3_chat', torch_dtype=torch_dtype)
        else:
            self.model = HPT1_5(global_model_path=model_id,
                                vis_scale=490,
                                prompt_template='phi3_chat', torch_dtype=torch_dtype)
        self.model.llm.to(device)
        self.model.visual_encoder.to(device)
        self.model.projector.to(device)

    def __call__(self, text, imgs):
        assert len(
            imgs) == 1, f'HPT does not support multi-image processing or pure text processing. Current number of images: {len(imgs)}'
        response = self.model.generate(
            prompt=text, image_path=imgs[0], dataset='demo')
        return response


load_from = 'modelscope'

# model_id = 'Qwen/Qwen2-0.5B-Instruct'
# model_id = 'Qwen/Qwen2-1.5B-Instruct'
# model_id = '/data/xiangyu/benchmarkModels/CHENCHEN/modelscope/hub/Qwen/Qwen2-7B-Instruct'
# model_id = 'Qwen/Qwen2-72B-Instruct'
# model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
# model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
# model_id = 'Qwen/Qwen2.5-3B-Instruct'
# model_id = '/data/xiangyu/benchmarkModels/CHENCHEN/modelscope/hub/Qwen/Qwen2___5-7B-Instruct'
# model_id = 'Qwen/Qwen2.5-32B-Instruct'
# model_id = 'Qwen/Qwen2.5-72B-Instruct'

model_id = "/data/xiangyu/benchmarkModels/CHENCHEN/huggingface/hub/models--HyperGAI--HPT1_5-Edge/snapshots/f5242c2767493723c48a8f62f9953ca320a43077"

model = HPT(model_id, device="cuda", fp16=True)


@app.route('/generate', methods=['POST'])
def generate_text():
    # Get prompt from the request
    data = request.json
    prompt = data.get("prompt", "")
    image_paths = data.get("image_paths", [])

    print(f"""
        This is the Llama-3___2-90B-Vision-Instruct.py script.
        The prompt is: {prompt}
        The image path is: {image_paths}
    """)

    images = []

    '''
    use a white image as placeholder if we have no images as input
    '''
    if len(image_paths) == 0:
        image_paths.append("/home/xiangyu/project/multimodalEDABenchmarking/models/MiniGPT-4/tmp.png")

    for i in range(len(image_paths)):
        image = Image.open(image_paths[i]).convert("RGB")
                # 获取图像的宽度和高度
        width, height = image.size
        # 计算新的尺寸，保持长边为512
        if width > height:
            new_width = 1024
            new_height = int((1024 / width) * height)
        else:
            new_height = 1024
            new_width = int((1024 / height) * width)

        # 调整图像大小
        resized_image = image.resize((new_width, new_height))

        images.append(resized_image)

    def combine_images(images):
        widths, heights = zip(*(i.size for i in images))

        max_width = max(widths)
        total_height = sum(heights)

        # Calculate the number of rows needed
        num_images = len(images)
        num_rows = (num_images + 1) // 2  # Each row has 2 images

        # Create a new image with the appropriate size
        combined_image = Image.new('RGB', (max_width * 2, max(heights) * num_rows))

        y_offset = 0
        for i in range(0, num_images, 2):
            x_offset = 0
            for j in range(2):
                if i + j < num_images:
                    combined_image.paste(images[i + j], (x_offset, y_offset))
                    x_offset += max_width
            y_offset += max(heights)

        # combined_image = combined_image.resize((512, 512))

        return combined_image

    # Combine images and use the combined image for processing
    # Qwen vl do not support images numbers > 1
    # it reuqires the image path as input, not the loaded images
    if len(images) > 1:
        combined_image = combine_images(images)
        combined_image_name = "combined_image_hpt_1_5_edge.png"
        combined_image.save(combined_image_name)
        combined_image_path = f"/home/xiangyu/project/multimodalEDABenchmarking/models/{combined_image_name}"
        images = [combined_image_path]
    else:
        images = image_paths

    response = model(prompt, imgs=images)

    print(response)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5027)