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

model_path = "/data/xiangyu/benchmarkModels/CHENCHEN/huggingface/hub/models--openbmb--MiniCPM3-4B/snapshots/e8a65f63cd4e4eff91571e603a2a34e50628ff67"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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

    messages = [
        {"role": "user", "content": prompt},
    ]
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

    model_outputs = model.generate(
        model_inputs,
        max_new_tokens=2048,
        top_p=0.7,
        temperature=0.7
    )

    output_token_ids = [
        model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
    ]

    responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
    print(responses)
    
    return jsonify(responses)
    
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5038)
