# update transformers using "pip install --upgrade transformers"
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

device = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = "/data/xiangyu/benchmarkModels/Llama-3___2-90B-Vision-Instruct"
# model_dir = snapshot_download(model_id, ignore_file_pattern=['*.pth'])
print(model_dir)

model = MllamaForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# model = model.to(device)

processor = AutoProcessor.from_pretrained(model_dir)

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
        images.append(image)

    # url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    content = []
    for i in range(len(images)):
        content.append({"type": "image"})
    
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "user", "content": content}
    ]
    
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images, input_text, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=1024)
    decoded_output = processor.decode(output[0])
    assistant_content = decoded_output.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1]
    assistant_content = assistant_content.strip()
    print(assistant_content)

    return jsonify(assistant_content)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5014)

