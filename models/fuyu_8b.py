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
from transformers import FuyuProcessor, FuyuForCausalLM

# from zmq import device

app = Flask(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_grad_enabled(False)

model_id = "/data/xiangyu/benchmarkModels/CHENCHEN/huggingface/hub/models--adept--fuyu-8b/snapshots/f41defefdb89be0d28cac19d94ce216e37cb6be5"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0")

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
    # if len(image_paths) == 0:
    #     image_paths.append("/home/xiangyu/project/multimodalEDABenchmarking/models/MiniGPT-4/tmp.png")

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
        images = [combined_image]
    else:
        pass

    inputs = processor(text=prompt, images=images, return_tensors="pt").to("cuda:0")
    # autoregressively generate text
    generation_output = model.generate(**inputs, max_new_tokens=1024)
    generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
    # assert generation_text == ['A blue bus parked on the side of a road.']

    generation_text = generation_text[0]
    print(generation_text)
    return jsonify(generation_text)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5026)