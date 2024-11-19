# env: benchmark-chenchen

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
import dashscope

app = Flask(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_grad_enabled(False)


class QWenVLWeb:
    def __init__(self, model_id, api_key):
        self.model_id = model_id
        dashscope.api_key = api_key

    def __call__(self, text, imgs):
        messages = [
            {
                'role': 'user',
                'content': []
            }
        ]
        for img in imgs:
            messages[0]['content'].append(
                {'image': f'file://{os.path.abspath(img)}'})
        messages[0]['content'].append({'text': text})
        try:
            response = dashscope.MultiModalConversation.call(
                model=self.model_id, messages=messages)
            return response['output']['choices'][0]['message']['content'][0]['text']
        except ValueError:
            return None

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

model_id = "qwen-vl-max"

api_key = 'sk-120232fab1ad45cb8b043be1c6c5517f'

model = QWenVLWeb(model_id, api_key)


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
    if model_id == 'qwen-vl-max':
        prompt += ' (you need to answer in English)'
    response = model(prompt, imgs=image_paths)

    # answer_start = response.find('</img>') + len('</img>')
    # answer = response[answer_start:].strip()
    print(response)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5023)