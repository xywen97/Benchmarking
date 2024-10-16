from .model_apis import query_llama2_7b, query_llama2_13b, query_llama3_8b_instruct, query_mistrial_7b, query_gpt35, query_gpt4o, query_gpt4
import json


def compute_acc(data):
    correct_count = 0
    total_count = 0
    comparison_results = {}

    original_data = data.get("original_data", {})
    answers = original_data.get("answers", [])
    predictions = data.get("answer_preds", [])
    
    for i, (answer, prediction) in enumerate(zip(answers, predictions)):
        key = f"question_{i}"
        comparison_results[key] = {
            "gt_answer": answer,
            "pred_answer": prediction,
            "is_correct": answer == prediction
        }
        if answer == prediction:
            correct_count += 1
        total_count += 1

    if total_count == 0:
        return 0.0, comparison_results

    accuracy = correct_count / total_count
    return accuracy, comparison_results


def choose_model(model_name):
    model_mapping = {
        "llama2_7b": query_llama2_7b,
        "llama2_13b": query_llama2_13b,
        "llama3_8b_instruct": query_llama3_8b_instruct,
        "mistrial_7b": query_mistrial_7b,
        "gpt35": query_gpt35,
        "gpt4o": query_gpt4o,
        "gpt4": query_gpt4
    }
    if model_name in model_mapping:
        return model_mapping[model_name]
    else:
        raise ValueError(f"Model '{model_name}' is not recognized. Please choose a valid model.")

def benchmarking_batch():
    pass

def benchmarking(raw_data, field="general", model='gpt4o'):
    # print(raw_data)
    if field.lower() not in ['architecture', 'backend', 'general', 'netlist', 'rtl', 'spec']:
        raise ValueError(f"Field '{field}' is not recognized. Please choose from 'architecture', 'backend', 'general', 'netlist', 'rtl', 'spec'.")
    
    data = raw_data[f"{field}_0"]

    statement_item = data['statement']
    image_item = data['images']
    question_item = data['questions']
    question_type_item = data['question_types']
    answer_item = data['answers']
    explanation_item = data['explanations']
    modality_item = data['modalities']
    difficulty_item = data['difficulties']
    ability_item = data['abilities']
    source_item = data['source']

    if not (len(question_item) == len(question_type_item) == len(answer_item) == len(explanation_item) == len(modality_item) == len(difficulty_item) == len(ability_item)):
        raise ValueError("Mismatch in the number of items: "
                         f"questions({len(question_item)}), "
                         f"question_types({len(question_type_item)}), "
                         f"answers({len(answer_item)}), "
                         f"explanations({len(explanation_item)}), "
                         f"modalities({len(modality_item)}), "
                         f"difficulties({len(difficulty_item)}), "
                         f"abilities({len(ability_item)})")


    
    print("Testing: ")
    print(f"This question is from {source_item}")
    
    query_model = choose_model(model.lower())


    answer_preds = []
    explanation_preds = []

    for i in range(len(question_item)):
        question_type = question_type_item[i]
        prompt_mapping = {
            "blank": """Filling in blanks question. Please answer in Json format and return Json object only. The response format should be: {{"answer": it should be a number/yes/no/not a sentence, "explanation": no more than 2 sentences for explanation on the answer}}""".strip(),
            "single": """Single choice question. Please answer in Json format and return Json object only. The response format should be: {{"answer": it should be a/b/c/d, "explanation": no more than 2 sentences for explanation on the answer}}"""
        }
        entire_question = statement_item + " " + question_item[i] + " " + prompt_mapping[question_type]
        print(entire_question)
        
        for attempt in range(3):
            try:
                response = query_model(entire_question)
                if response:
                    print(f"Response received: {response}")
                    try:
                        # 尝试将响应解析为JSON对象
                        # 处理响应中包含的json``` ```的情况
                        if response.startswith("```json") and response.endswith("```"):
                            response = response[7:-3].strip()
                        print(response)
                        response_data = json.loads(response)
                        answer_pred = response_data.get("answer", "No answer found")
                        explanation_pred = response_data.get("explanation", "No explanation found")
                        print(f"Extracted Answer: {answer_pred}")
                        print(f"Extracted Explanation: {explanation_pred}")
                        answer_preds.append(str(answer_pred))
                        explanation_preds.append(explanation_pred)
                        break
                    except json.JSONDecodeError:
                        print("Failed to decode JSON from response.")
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
        else:
            print("All attempts to query the model failed.")
        
    returned_response = {
        "original_data": {
            "statement": statement_item,
            "questions": question_item,
            "question_types": question_type_item,
            "answers": answer_item,
            "explanations": explanation_item,
            "modalities": modality_item,
            "difficulties": difficulty_item,
            "abilities": ability_item,
            "source": source_item
        },
        "answer_preds": answer_preds,
        "explanation_preds": explanation_preds
    }

    return returned_response