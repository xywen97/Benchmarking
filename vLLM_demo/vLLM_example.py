# Import necessary libraries
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from dataloader import load_data
from Benchmarking.vLLM_demo.vision_language_testing_script import run_vision_language

# set cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
use_vllm = True

# Define model ID and paths
model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
# model_path = '/data/xiangyu/benchmarkModels/CHENCHEN/modelscope/hub/Qwen/Qwen2___5-7B-Instruct'
model_path = "If you cannot download the model, you can put the model_path here, and replace the model_id with the model_path"

try:
    # Attempt to load the model using VLLM
    print("Attempting to use VLLM...")
    # 
    model = LLM(model=model_id, tensor_parallel_size=1)
    vllm_available = True
    print("VLLM is available and in use.")
except Exception as e:
    # If VLLM is not available, print the error message
    print(f"VLLM not available: {e}")


def choose_template(model_name):
    chatML_template_qwen = """<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}
"""
    if "Qwen" in model_name:
        return chatML_template_qwen
    else:
        return chatML_template_qwen

def prepare_prompt(questions, model_name):
    prompt = choose_template(model_name)
    prompts = [prompt.format(
        f"{question}",
        ""
    ) for question in questions]
    # print(f"prompts: {prompts}")
    return prompts

def generate_text(questions, model_name):
    print("Generating text using VLLM...")
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1024)
    # we must prepare the prompt here, otherwise the generation will not follow the instruction
    prompts = prepare_prompt(questions, model_name)
    outputs = model.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    return outputs

# Example usage
if __name__ == "__main__":
    questions = ["What is the capital of France?", "What is the capital of Germany?"]
    response = generate_text(questions, model_id)
    print(f"Response: {response}") 