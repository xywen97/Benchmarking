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
    data_path = "data/tmp.json"
    image_path = "data/tmp"
    data = load_data(data_path, image_path)

    '''
    benchmarking the model using one question
    '''
    field = "general"
    # available models:
    # llama2_7b, llama2_13b, llama3_8b_instruct, mistrial_7b, gpt35, gpt4o, gpt4, gpt4_turbo, gpt4v
    return_data = benchmarking(data, field, model_name='gpt35')
    results, comparisons = compute_acc(return_data)

    print(f"results: {results}")
    print(json.dumps({"comparison": comparisons}, ensure_ascii=False, indent=4))
    '''
    benchmarking the model using batch of questions
    '''
    # benchmarking_batch(data)
