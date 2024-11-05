from utils.parser import benchmarking, benchmarking_batch, compute_acc
from utils.dataloader import load_data, load_data_batch
import json
import argparse
from utils.parser import benchmarking, compute_acc
from utils.dataloader import load_data
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Benchmarking script for model evaluation.")
    parser.add_argument('--field', type=str, required=True, help='Field to benchmark, e.g., spec, general, etc.')
    parser.add_argument('--model', type=str, required=True, help='Model to use, e.g., gpt4o, llama2_7b, etc.')
    args = parser.parse_args()

    data_path = f"data/{args.field}.json"
    image_path = f"data/{args.field}"
    data = load_data(data_path, image_path)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"./results/all_results_{args.field}_{args.model}_{current_time}.jsonl"
    
    return_data = benchmarking(data, args.field, model_name=args.model, save_path=save_path)

    # with open(f"./results/all_results_{args.field}_{args.model}.json", 'w', encoding='utf-8') as f:
    #     json.dump(return_data, f, ensure_ascii=False, indent=4)

    results, comparisons = compute_acc(return_data)

    with open(f"./results/accuracy_{args.field}_{args.model}_{current_time}.json", 'w', encoding='utf-8') as f:
        json.dump({"Accuracy": results, "Comparison": comparisons}, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
