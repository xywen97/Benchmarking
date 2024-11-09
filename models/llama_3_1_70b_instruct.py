from flask import Flask, request, jsonify
import torch
from transformers import pipeline
from transformers import AutoTokenizer
import os

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model_path = "/data/xiangyu/benchmarkModels/Meta-Llama-3___1-70B-Instruct"

model = pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


terminators = [
    model.tokenizer.eos_token_id,
    model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

@app.route('/generate', methods=['POST'])
def generate_text():
    # Get prompt from the request
    data = request.json
    prompt = data.get("prompt", "")

    print(f"""
        This is the llama_3_70b_instruct.py script.
        The prompt is: {prompt}
    """)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = model(
        messages,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
    )

    # print(response)
    response = response[0]['generated_text'][-1]['content']

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5011)
