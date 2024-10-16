from flask import Flask, request, jsonify
import torch
from transformers import pipeline
from transformers import AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

app = Flask(__name__)

# Load your model and tokenizer
model_path = "/data/zyzheng23/xiangyu/hf_ckpts/llama_2_13b_chat_hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

@app.route('/generate', methods=['POST'])
def generate_text():
    # Get prompt from the request
    data = request.json
    prompt = data.get("prompt", "")

    print(f"""
        This is the llama_2_13b_chat_hf.py script.
        The prompt is: {prompt}
    """)
    
    prompt = [
        # {"role": "system", "content": "You are math assistant. You can help with math problems."},
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    # Generate text
    response = model(
        prompt, 
        do_sample=True, 
        top_k=10, 
        num_return_sequences=1, 
        eos_token_id=tokenizer.eos_token_id,
        truncation=True, 
        max_length=512,
        temperature=0.7,
    )

    response = response[0]['generated_text'][-1]['content']
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5002)
