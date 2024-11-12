from flask import Flask, request, jsonify
import torch
from transformers import pipeline
from transformers import AutoTokenizer
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = "/data/xiangyu/benchmarkModels/CHENCHEN/huggingface/hub/models--internlm--internlm-chat-7b/snapshots/4e2d2e185058a4cbaa2377ba1bb993eb599ce492"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()

@app.route('/generate', methods=['POST'])
def generate_text():
    # Get prompt from the request
    data = request.json
    prompt = data.get("prompt", "")

    print(f"""
        This is the llama_3_8b_instruct.py script.
        The prompt is: {prompt}
    """)

    response, history = model.chat(tokenizer, prompt, history=[])
    print(response)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5028)
