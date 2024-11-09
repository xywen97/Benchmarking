import os
import jsonlines
import json
from utils.parser import benchmarking, compute_acc

def read_multiline_json(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        buffer = ""
        for line in file:
            line = line.strip()
            if line:  # 忽略空行
                buffer += line
                try:
                    # 尝试解析 JSON 对象
                    json_object = json.loads(buffer)
                    # print(json_object)
                    if list(json_object.values())[0] == "None":
                        buffer = ""
                        continue
                    data[list(json_object.keys())[0]] = list(json_object.values())[0]
                    buffer = ""  # 清空缓冲区以准备下一个 JSON 对象
                except json.JSONDecodeError:
                    # 如果解析失败，继续累积行
                    continue
    return data

if __name__ == "__main__":
    base_path = "/home/xiangyu/project/multimodalEDABenchmarking/results/"
    file_name = "all_results_rtl_llama2_13b.jsonl"
    file_path = base_path + file_name
    
    data = read_multiline_json(file_path)


    results, comparisons = compute_acc(data)

    save_path = base_path + "accuracy_" + file_name[4:].replace('jsonl', 'json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({"Accuracy": results, "Comparison": comparisons}, f, ensure_ascii=False, indent=4)