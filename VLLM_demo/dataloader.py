import json

def load_data_batch(field="general", base_path="data"):
    # 定义图像文件夹路径和JSON文件路径
    image_folder = f"{base_path}/{field}"
    json_file = f"{base_path}/{field}.json"
    
    # 读取并解析JSON文件
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 解析JSON数据
    parsed_data = {}
    for key, value in data.items():
        parsed_data[key] = {
            "statement": value.get("statement", ""),
            "images": [f"{image_folder}/{img}" for img in value.get("image", [])],
            "questions": value.get("question", []),
            "question_types": value.get("question_type", []),
            "answers": value.get("answer", []),
            "explanations": value.get("explanation", []),
            "modalities": value.get("modality", []),
            "difficulties": value.get("difficulty", []),
            "abilities": value.get("ability", []),
            "source": value.get("source", "")
        }
    
    # 返回解析后的数据
    return parsed_data

def load_data(datapath, imagepath, start, end):    
    # 读取并解析JSON文件
    with open(datapath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if end == -1:
        end = len(data.keys()) - 1

    # 解析JSON数据
    parsed_data = {}
    for key, value in data.items():
        item_index = key.split('_')[1]
        item_index = int(item_index)
        if item_index >= start and item_index <= end:
            parsed_data[key] = {
                "statement": value.get("statement", ""),
                "images": [f"{imagepath}/{img}" for img in value.get("image", [])],
                "questions": value.get("question", []),
                "question_types": value.get("question_type", []),
                "answers": value.get("answer", []),
                "explanations": value.get("explanation", []),
                "modalities": value.get("modality", []),
                "difficulties": value.get("difficulty", []),
                "abilities": value.get("ability", []),
                "source": value.get("source", "")
            }
        else:
            pass
    
    # 返回解析后的数据
    return parsed_data