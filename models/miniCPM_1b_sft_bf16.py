# Transformers: 4.36/4.37
# Environment: minigpt4v

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
from flask import Flask, request, jsonify
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

model_path = "/data/xiangyu/benchmarkModels/CHENCHEN/huggingface/hub/models--openbmb--MiniCPM-1B-sft-bf16/snapshots/85252935bfc9acb3ee93dcea4b793fb87a0b43e8"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

model = model.to(device=device, dtype=torch.bfloat16)
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

    responses, history = model.chat(tokenizer, prompt, temperature=0.8, top_p=0.8)

    print(responses)
    
    return jsonify(responses)
    
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5039)
