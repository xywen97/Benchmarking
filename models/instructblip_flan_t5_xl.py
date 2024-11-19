# env: benchmark

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

model = InstructBlipForConditionalGeneration.from_pretrained("/data/xiangyu/benchmarkModels/instructblip-flan-t5-xl")
processor = InstructBlipProcessor.from_pretrained("/data/xiangyu/benchmarkModels/instructblip-flan-t5-xl")

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


@app.route('/generate', methods=['POST'])
def generate_text():
    # Get prompt from the request
    data = request.json
    prompt = data.get("prompt", "")
    image_paths = data.get("image_paths", [])

    print(f"""
        This is the instructblip_flan_t5_xl.py script.
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
            new_width = 512
            new_height = int((512 / width) * height)
        else:
            new_height = 512
            new_width = int((512 / height) * width)

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
    combined_image = combine_images(images)
    # combined_image.save("combined_image.png")
    images = [combined_image]
    
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=1024,
            min_length=10,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
    )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    
    print(generated_text)
    return jsonify(generated_text)
    
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5005)
