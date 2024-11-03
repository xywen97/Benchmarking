from utils.parser import benchmarking, benchmarking_batch, compute_acc
from utils.dataloader import load_data, load_data_batch
import json

if __name__ == "__main__":
    '''
    load data in batch from the data folder by fields
    '''
    # field = "general"
    # data = load_data_batch(field=field)
    
    '''
    load test example for testing
    '''
    data_path = "data/spec.json"
    image_path = "data/spec"
    data = load_data(data_path, image_path)

    '''
    benchmarking the model using one question
    '''
    field = "spec"
    # available models:
    # llama2_7b, llama2_13b, llama3_8b_instruct, mistrial_7b, gpt35, gpt4o, gpt4, gpt4_turbo, gpt4v
    return_data = benchmarking(data, field, model_name='gpt4o')

    with open(f"{field}.json", 'w', encoding='utf-8') as f:
        json.dump(return_data, f, ensure_ascii=False, indent=4)

    results, comparisons = compute_acc(return_data)

    with open(f"results_and_comparisons_{field}.json", 'w', encoding='utf-8') as f:
        json.dump({f"Accuracy": results, "Comparison": comparisons}, f, ensure_ascii=False, indent=4)
    
    '''
    benchmarking the model using batch of questions
    '''
    # benchmarking_batch(data)
