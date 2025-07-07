# Import necessary libraries
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from dataloader import load_data
from vision_language import run_vision_language

# set cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
use_vllm = True

# Define model ID and paths
model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
# model_path = '/data/xiangyu/benchmarkModels/CHENCHEN/modelscope/hub/Qwen/Qwen2___5-7B-Instruct'
model_path = "If you cannot download the model, you can put the model_path here, and replace the model_id with the model_path"

# Check if VLLM can be used
if use_vllm:
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
else:
    vllm_available = False

if not use_vllm or not vllm_available:
    # Load model from Hugging Face or ModelScope
    try:
        # Attempt to load the model and tokenizer from Hugging Face
        print("Loading model from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto', trust_remote_code=True)
        print("Model loaded from Hugging Face.")
    except Exception as e:
        # If loading from Hugging Face fails, try ModelScope
        print(f"Failed to load from Hugging Face: {e}")
        print("Attempting to load from ModelScope...")
        from modelscope import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto', trust_remote_code=True)
        print("Model loaded from ModelScope.")


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


# Define function to generate text
def generate_text(questions, model_name):
    if use_vllm and vllm_available:
        # Generate text using VLLM
        print("Generating text using VLLM...")
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1024)
        # we must prepare the prompt here, otherwise the generation will not follow the instruction
        prompts = prepare_prompt(questions, model_name)
        outputs = model.generate(prompts, sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs
    else:
        # Generate text using the standard method
        print("Generating text using standard method...")
        # we must prepare the prompt here, otherwise the generation will not follow the instruction
        prompts = prepare_prompt(questions, model_name)
        # Enable padding and truncation
        model_inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024)
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return responses

# Example usage
if __name__ == "__main__":
    # question = "What is the capital of France?"
    # response = generate_text(question, model_id)
    # print(f"Response: {response}") 

    data_path = f"../data/backend.json"
    image_path = f"../data/backend"
    batch_instance_size = 10
    data = load_data(data_path, image_path, 0, -1)
    # convert the data to a list
    data = list(data.values())
    # convert the data to a list of questions
    for i in range(0, len(data), batch_instance_size):
        batch_data = data[i:i+batch_instance_size]
        # questions = [item['questions'] for item in batch_data]
        questions = []
        for item in batch_data:
            questions.extend(item['questions'])
        print(f"questions: {questions}")
        
        outputs = generate_text(questions, model_id)
        print(f"Response: {outputs}") 
        break

    # run the vision language model
    print("Running vision language model...")
    run_vision_language()