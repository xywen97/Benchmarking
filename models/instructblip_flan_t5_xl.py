from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

model = InstructBlipForConditionalGeneration.from_pretrained("/data/xiangyu/benchmarkModels/instructblip-flan-t5-xl")
processor = InstructBlipProcessor.from_pretrained("/data/xiangyu/benchmarkModels/instructblip-flan-t5-xl")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
        images.append(image)
    images = images[:1]
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=512,
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
