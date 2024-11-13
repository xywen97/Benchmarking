# Transformers: 4.36/4.37
# Environment: minigpt4v

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import requests
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

model_path = "/data/xiangyu/benchmarkModels/CHENCHEN/huggingface/hub/models--openbmb--MiniCPM-V-2/snapshots/ee00ff7ce36667e7df81cb2a018951b663bdcc59"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device=device, dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.eval()

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
        image = image.resize((512, 512))
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

        combined_image = combined_image.resize((512, 512))

        return combined_image

    # Combine images and use the combined image for processing
    if len(images) > 1:
        combined_image = combine_images(images)
        # combined_image.save("combined_image.png")
        images = [combined_image]
    else:
        pass
    
    msgs = [{'role': 'user', 'content': prompt}]

    res, context, _ = model.chat(
        image=images[0],
        msgs=msgs,
        context=None,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7
    )
    print(res)
    
    return jsonify(res)
    
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5037)
