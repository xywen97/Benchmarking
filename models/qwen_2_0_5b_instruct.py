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

from zmq import device

app = Flask(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_grad_enabled(False)


class QWen:
    def __init__(self, model_id, device, fp16, load_from='huggingface'):
        assert load_from in [
            'huggingface', 'modelscope'], 'invalid model source. valid sources are: `huggingface`, `modelscope`'
        if load_from == 'huggingface':
            from transformers import AutoModelForCausalLM, AutoTokenizer
        else:
            from modelscope import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True)
        torch_dtype = torch.float16 if fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            device_map=device, trust_remote_code=True)

    # single-modal. `img` is unused
    def __call__(self, text, imgs):
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': text}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(
            [prompt], return_tensors='pt').to(self.model.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids, max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in
            zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        return response


load_from = 'modelscope'

model_id = '/data/xiangyu/benchmarkModels/CHENCHEN/modelscope/hub/Qwen/Qwen2-0___5B-Instruct'
# model_id = 'Qwen/Qwen2-1.5B-Instruct'
# model_id = '/data/xiangyu/benchmarkModels/CHENCHEN/modelscope/hub/Qwen/Qwen2-7B-Instruct'
# model_id = 'Qwen/Qwen2-72B-Instruct'
# model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
# model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
# model_id = 'Qwen/Qwen2.5-3B-Instruct'
# model_id = '/data/xiangyu/benchmarkModels/CHENCHEN/modelscope/hub/Qwen/Qwen2___5-7B-Instruct'
# model_id = 'Qwen/Qwen2.5-32B-Instruct'
# model_id = 'Qwen/Qwen2.5-72B-Instruct'

model = QWen(model_id, device="auto", fp16=True, load_from=load_from)


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

    response = model(prompt, imgs=None)

    print(response)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5020)