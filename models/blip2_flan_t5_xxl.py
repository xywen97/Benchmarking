# env: benchmark

from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
import torch
from PIL import Image
import requests
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

processor = AutoProcessor.from_pretrained("/data/xiangyu/benchmarkModels/blip2-flan-t5-xxl")
model = AutoModelForVisualQuestionAnswering.from_pretrained("/data/xiangyu/benchmarkModels/blip2-flan-t5-xxl")

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


@app.route('/generate', methods=['POST'])
def generate_text():
    # Get prompt from the request
    data = request.json
    prompt = data.get("prompt", "")
    image_paths = data.get("image_paths", [])

    print(f"""
        This is the blip2_flan_t5_xxl.py script.
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
        images.append(image)
        
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
            max_length=2048,
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
    app.run(debug=False, host='0.0.0.0', port=5008)
