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

app = Flask(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_grad_enabled(False)


class Bunny:
    def __init__(self, model_id, device, fp16, load_from='huggingface'):
        assert load_from in [
            'huggingface', 'modelscope'], 'invalid model source. valid sources are: `huggingface`, `modelscope`'
        if load_from == 'huggingface':
            from transformers import AutoModelForCausalLM, AutoTokenizer
        else:
            from modelscope import AutoModelForCausalLM, AutoTokenizer
        self.offset_bos = 0 if 'Bunny-v1_0-3B' in model_id else 1
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True)
        torch_dtype = torch.float16 if fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype,
            low_cpu_mem_usage=True, device_map=device, trust_remote_code=True)
        statement = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: '
        self.statement_id = self.tokenizer(statement).input_ids

    def __call__(self, text, imgs):
        assert len(imgs) > 0, 'Bunny does not support pure text processing'
        text += ' ASSISTANT:'
        prompt_id = self.tokenizer(text).input_ids
        input_ids = torch.tensor(
            self.statement_id + [-200] + prompt_id[self.offset_bos:], dtype=torch.long).unsqueeze(0).to(self.model.device)
        imgs = [Image.open(img).convert('RGB') for img in imgs]
        imgs = self.model.process_images(imgs, self.model.config).to(
            dtype=self.model.dtype, device=self.model.device)
        input_ids = input_ids.to(self.model.device)
        # Ensure all tensors are on the same device
        imgs = imgs.to(self.model.device)
        output_ids = self.model.generate(
            input_ids, images=imgs, max_new_tokens=1024, use_cache=True,
            repetition_penalty=1.0)[0]
        return self.tokenizer.decode(
            output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()


load_from = 'modelscope'

model_id = "/data/xiangyu/benchmarkModels/CHENCHEN/modelscope/hub/BAAI/Bunny-v1_0-3B"

model = Bunny(model_id, device='cuda', fp16=True, load_from=load_from)


@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get("prompt", "")
    image_paths = data.get("image_paths", [])

    print(f"""
        This is the Llama-3___2-90B-Vision-Instruct.py script.
        The prompt is: {prompt}
        The image path is: {image_paths}
    """)

    response = model(prompt, imgs=image_paths)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5025)