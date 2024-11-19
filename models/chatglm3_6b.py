# pip install transformers==4.41.2
# env: benchmark-liuyi

from modelscope import AutoTokenizer, AutoModel, snapshot_download
import torch
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 设置可见设备为GPU 0

device = "cuda" if torch.cuda.is_available() else "cpu"


# model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision = "v1.0.0")
model_dir = "/data/xiangyu/benchmarkModels/chatglm3-6b"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
model = model.eval()

@app.route('/generate', methods=['POST'])
def generate_text():
    # Get prompt from the request
    data = request.json
    prompt = data.get("prompt", "")

    print(f"""
        This is the chatglm3_6b.py script.
        The prompt is: {prompt}
    """)
    
    # prompt = [
    #     # {"role": "system", "content": "You are math assistant. You can help with math problems."},
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": prompt},
    # ]

    print(prompt)

    response, history = model.chat(tokenizer, prompt, history=[])
    print(response)
    # response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    # print(response)

    # response = response[0]['generated_text'][-1]['content']
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5009)