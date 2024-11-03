from utils.parser import benchmarking, benchmarking_batch, compute_acc
from utils.dataloader import load_data, load_data_batch
import json
import argparse
from utils.parser import benchmarking, compute_acc
from utils.dataloader import load_data

def main():
    parser = argparse.ArgumentParser(description="Benchmarking script for model evaluation.")
    parser.add_argument('--field', type=str, required=True, help='Field to benchmark, e.g., spec, general, etc.')
    parser.add_argument('--model', type=str, required=True, help='Model to use, e.g., gpt4o, llama2_7b, etc.')
    args = parser.parse_args()

    data_path = f"data/{args.field}.json"
    image_path = f"data/{args.field}"
    data = load_data(data_path, image_path)

    return_data = benchmarking(data, args.field, model_name=args.model)

    with open(f"all_results_{args.field}_{args.model}.json", 'w', encoding='utf-8') as f:
        json.dump(return_data, f, ensure_ascii=False, indent=4)

    results, comparisons = compute_acc(return_data)

    with open(f"accuracy_{args.field}_{args.model}.json", 'w', encoding='utf-8') as f:
        json.dump({"Accuracy": results, "Comparison": comparisons}, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
