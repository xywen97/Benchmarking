from pyexpat import model
from .model_apis import query_llama2_7b, query_llama2_13b, query_llama3_8b_instruct, query_mistrial_7b, query_gpt35, query_gpt4o, query_gpt4, query_gpt4_turbo, query_gpt4v, image_captioning, query_minigpt4_vicuna7b, query_gemini_series, query_claude_series, query_reka_series, query_instructblip_flan_t5_xl, query_instructblip_flan_t5_xxl, query_blip2_flan_t5_xl, query_blip2_flan_t5_xxl, query_chatglm3_6b, query_llama3_1_70b_instruct, query_llama3_1_8b_instruct, query_llama3_2_11b_vision_instruct, query_llama3_2_90b_vision_instruct, query_kosmos_2_patch14_224, query_qwen_2_5_7b_instruct, query_qwen_2_5_72b_instruct, query_qwen_2_7b_instruct, query_qwen_2_72b_instruct, query_qwen_2_0_5b_instruct, query_qwen_vl
import json
import base64
import time

data_image_base_path = "/home/xiangyu/project/multimodalEDABenchmarking/"

def compute_acc(data):
    correct_count = 0
    total_count = 0
    comparison_results = {}

    for key in data.keys():
        current_data = data[key]
        # original_data = data.get("original_data", {})
        answers = current_data.get("answer", [])
        predictions = current_data.get("answer_preds", [])
        
        acc_per_question_total_count = 0
        acc_per_question_correct_count = 0

        current_comparison_results = {}

        for i, (answer, prediction) in enumerate(zip(answers, predictions)):
            question_key = f"question_{i}"
            current_comparison_results[question_key] = {
                "gt_answer": answer,
                "pred_answer": prediction,
                "is_correct": answer.lower() == prediction.lower()
            }
            if answer == prediction:
                correct_count += 1
                acc_per_question_correct_count += 1
            total_count += 1
            acc_per_question_total_count += 1
        
        acc_per_question = acc_per_question_correct_count/acc_per_question_total_count if acc_per_question_total_count != 0 else 0
        current_comparison_results["accuracy"] = acc_per_question
        comparison_results[key] = current_comparison_results

    if total_count == 0:
        return 0.0, comparison_results

    accuracy = correct_count / total_count
    return accuracy, comparison_results


def choose_model(model_name):
    if model_name.startswith('gemini'):
        return query_gemini_series
    if model_name.startswith('claude'):
        return query_claude_series
    if model_name.startswith('reka'):
        return query_reka_series
    model_mapping = {
        "llama2_7b": query_llama2_7b,
        "llama2_13b": query_llama2_13b,
        "llama3_8b_instruct": query_llama3_8b_instruct,
        "llama3_1_8b_instruct": query_llama3_1_8b_instruct,
        "llama3_1_70b_instruct": query_llama3_1_70b_instruct,
        "llama3_2_11b_vision_instruct": query_llama3_2_11b_vision_instruct,
        "llama3_2_90b_vision_instruct": query_llama3_2_90b_vision_instruct,
        "mistrial_7b": query_mistrial_7b,
        "gpt35": query_gpt35,
        "gpt4o": query_gpt4o,
        "gpt4": query_gpt4,
        "gpt4_turbo": query_gpt4_turbo,
        "gpt4v": query_gpt4v,
        "minigpt4_vicuna7b": query_minigpt4_vicuna7b,
        "instructblip_flan_t5_xl": query_instructblip_flan_t5_xl,
        "instructblip_flan_t5_xxl": query_instructblip_flan_t5_xxl,
        "blip2_flan_t5_xl": query_blip2_flan_t5_xl,
        "blip2_flan_t5_xxl": query_blip2_flan_t5_xxl,
        "chatglm3_6b": query_chatglm3_6b,
        "kosmos2_patch14_224": query_kosmos_2_patch14_224,
        "qwen_2_5_7b_instruct": query_qwen_2_5_7b_instruct,
        "qwen_2_5_72b_instruct": query_qwen_2_5_72b_instruct,
        "qwen_2_7b_instruct": query_qwen_2_7b_instruct,
        "qwen_2_72b_instruct": query_qwen_2_72b_instruct,
        "qwen_2_0_5b_instruct": query_qwen_2_0_5b_instruct,
        "qwen_vl": query_qwen_vl
    }
    if model_name in model_mapping:
        return model_mapping[model_name]
    else:
        raise ValueError(f"Model '{model_name}' is not recognized. Please choose a valid model.")

def check_if_multi_modal(model_name):
    if model_name.startswith('gemini') or model_name.startswith('claude'):
        return False
    
    if model_name.startswith('reka'):
        return "url_base64"

    is_multi_mapping = {
        "llama2_7b": False,
        "llama2_13b": False,
        "llama3_8b_instruct": False,
        "llama3_1_8b_instruct": False,
        "llama3_1_70b_instruct": False,
        "mistrial_7b": False,
        "gpt35": False,
        "gpt4o": "type_base64",
        "gpt4": "type_base64",
        "gpt4_turbo": "type_base64",
        "gpt4v": "type_base64",
        "minigpt4_vicuna7b": "type_raw",
        "instructblip_flan_t5_xl": "type_raw",
        "instructblip_flan_t5_xxl": "type_raw",
        "blip2_flan_t5_xl": "type_raw",
        "blip2_flan_t5_xxl": "type_raw",
        "chatglm3_6b": False,
        "llama3_2_11b_vision_instruct": "type_raw",
        "llama3_2_90b_vision_instruct": "type_raw",
        "kosmos2_patch14_224": "type_raw",
        "qwen_2_5_7b_instruct": False,
        "qwen_2_5_72b_instruct": False,
        "qwen_2_7b_instruct": False,
        "qwen_2_72b_instruct": False,
        "qwen_2_0_5b_instruct": False,
        "qwen_vl": "type_raw"
    }

    return is_multi_mapping[model_name]

def load_image_base64(image_path):
    """
    Encode an image file to Base64

    Input:
         - image_path: The file path of the image
    Output:
         - The encoded Base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def benchmarking_batch():
    pass

def benchmarking(raw_data, field="general", model_name='gpt4o', save_path=None):
    if save_path is None:
        raise ValueError("The save_path cannot be None. Please provide a valid path to save the results.")

    if field.lower() not in ['architecture', 'backend', 'general', 'netlist', 'rtl', 'spec', 'tmp']:
        raise ValueError(f"Field '{field}' is not recognized. Please choose from 'architecture', 'backend', 'general', 'netlist', 'rtl', 'spec'.")
    
    field_responses = {}

    for key in raw_data.keys():
        data = raw_data[key]

        # data = raw_data[f"{field}_0"]

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

        '''
        Check if the elements in each item is full filled
        '''
        # if not (len(question_item) == len(question_type_item) == len(answer_item) == len(explanation_item) == len(modality_item) == len(difficulty_item) == len(ability_item)):
        if not (len(question_item) == len(answer_item)):
            print(f"Mismatch in the number of items of question {key}: "
                  f"questions({len(question_item)}), "
                  f"question_types({len(question_type_item)}), "
                  f"answers({len(answer_item)}), "
                  f"explanations({len(explanation_item)}), "
                  f"modalities({len(modality_item)}), "
                  f"difficulties({len(difficulty_item)}), "
                  f"abilities({len(ability_item)})")

            returned_response = "None"
            with open(save_path, 'a', encoding='utf-8') as f:
                json.dump({key: returned_response}, f, ensure_ascii=False, indent=4)
                f.write('\n')
            
            continue

        line_length = 50
        print("#" * line_length)
        print(f"# We are now benchmarking {model_name}...".ljust(line_length - 1) + "#")
        print(f"# This question is from {source_item}...".ljust(line_length - 1) + "#")
        print(f"# This is QUESTION: {key} for this field...".ljust(line_length - 1) + "#")
        print("#" * line_length)
        print()
        
        '''
        Choose candidate models
        '''
        query_model = choose_model(model_name.lower())

        '''
        Determine if the model supports multi-modal reasoning
        '''
        is_multi_modal = check_if_multi_modal(model_name)

        '''
        Load images correspondingly
        '''
        images = []
        image_captions = []
        if not is_multi_modal:
            for image_name in image_item:
                try:
                    print("### Image captioning...")
                    image_caption = image_captioning(image_path=image_name)
                    print(f"extracted image caption: {image_caption['caption']}")
                    image_captions.append(image_caption['caption'])
                except Exception as e:
                    print(f"An error occurred while generating image caption: {e}")
            
        elif is_multi_modal == "type_base64":
            for image_name in image_item:
                try:
                    image = load_image_base64(image_name)
                    print(f"load image: {image_name}")
                    images.append(image)
                except FileNotFoundError:
                    print(f"Image {image_name} not found.")
        
        elif is_multi_modal == "type_raw":
            for image_name in image_item:
                try:
                    print(f"preparing raw image paths: {image_name}")
                    images.append(data_image_base_path + image_name)
                except Exception as e:
                    print(f"An error occurred while preparing raw image paths: {e}")
        
        elif is_multi_modal == "url_base64":
            for image_name in image_item:
                try:
                    image = load_image_base64(image_name)
                    print(f"load image: {image_name}")
                    images.append(f"data:image/jpeg;base64,{image}")
                except FileNotFoundError:
                    print(f"Image {image_name} not found.")

        else:
            pass
        
        print()

        '''
        Query model for getting answers
        '''
        answer_preds = []
        explanation_preds = []
        raw_preds = []

        # print(f"STATEMENT: {statement_item}")
        # print()
        for i in range(len(question_item)):
            question_type = question_type_item[i]
            prompt_mapping = {
                # "blank": """Filling in blanks question. Please answer in Json format and return Json object only. The response format should be: {{"answer": it should be a number/yes/no/not a sentence, "explanation": no more than 3 sentences for explanation on your thought to give the answer.}}""".strip(),
                # "single": """Single choice question. Please answer in Json format and return Json object only. The response format should be: {{"answer": it should be a/b/c/d, "explanation": no more than 3 sentences for explanation on your thought to give the answer.}}""".strip(),
            }

            if question_type not in prompt_mapping.keys():
                if model_name in ['instructblip_flan_t5_xl', 'instructblip_flan_t5_xxl', 'blip2_flan_t5_xl', 'blip2_flan_t5_xxl', 'kosmos2_patch14_224']:
                    prompt_mapping[question_type] = """
                        Question:
                    """.strip()
                else:
                    prompt_mapping[question_type] = """
                        Answer this question in Json format and return Json object only. The response format should be {{"answer": your answer to this question, "explanation": no more than 3 sentences for explanation on your thought to give the answer.}}
                    """.strip()


            image_captions_hint = ",".join(image_captions)

            if not is_multi_modal and len(image_captions) > 0:
                # use the image captions as the additional cues for answering this question
                entire_question = statement_item + " " + question_item[i] + " " + prompt_mapping[question_type] + " You may refer to the cues that extracted from several provided images: " + image_captions_hint 
            elif not is_multi_modal and len(image_captions) == 0:
                entire_question = statement_item + " " + question_item[i] + " " + prompt_mapping[question_type]
            elif is_multi_modal:
                if model_name in ['instructblip_flan_t5_xl', 'instructblip_flan_t5_xxl', 'blip2_flan_t5_xl', 'blip2_flan_t5_xxl', 'kosmos2_patch14_224']:
                    entire_question = prompt_mapping[question_type] + " " + statement_item + " " + question_item[i] + " " + "Answer:"
                else:
                    if len(images) == 0:
                        entire_question = statement_item + " " + question_item[i] + " " + prompt_mapping[question_type]
                    else:
                        entire_question = statement_item + " " + question_item[i] + " " + prompt_mapping[question_type] + " You may refer to the provided images."
                
            else:
                print("not handled yet..")
                pass
            
            print(f"QUESTION: {entire_question}")
            
            for attempt in range(3):
                try:
                    if model_name.startswith("gemini") or model_name.startswith("claude"):
                        response = query_model(entire_question, images, model_name=model_name)
                        time.sleep(15)
                    elif model_name.startswith("reka"):
                        response = query_model(entire_question, images, model_name=model_name)
                    else:
                        response = query_model(entire_question, images)
                    if response:
                        # print(f"Response received: {response}")
                        try:
                            # 尝试将响应解析为JSON对象
                            # 处理响应中包含的json``` ```的情况
                            response = response.strip()
                            response = response.replace("\n", "")
                            if response.startswith("```json") and response.endswith("```"):
                                response = response[7:-3].strip()
                            # print(response)
                            response_data = json.loads(response)
                            answer_pred = response_data.get("answer", "No answer found")
                            explanation_pred = response_data.get("explanation", "No explanation found")
                            print(f"--> Extracted Answer: {answer_pred}")
                            print(f"--> Extracted Explanation: {explanation_pred}")
                            answer_preds.append(str(answer_pred))
                            explanation_preds.append(explanation_pred)
                            raw_preds.append("None")
                            break
                        except json.JSONDecodeError:
                            print("Failed to decode JSON from response.")
                            if attempt == 2:
                                answer_preds.append("None")
                                explanation_preds.append("None")
                                raw_preds.append(response)
                                print("Appending raw data...")
                            
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed with error: {e}")
            else:
                print("All attempts to query the model failed.")
            
            print("#" * 100)
            print()
            
        # returned_response = {
        #     "original_data": {
        #         "statement": statement_item,
        #         "questions": question_item,
        #         "question_types": question_type_item,
        #         "answers": answer_item,
        #         "explanations": explanation_item,
        #         "modalities": modality_item,
        #         "difficulties": difficulty_item,
        #         "abilities": ability_item,
        #         "source": source_item
        #     },
        #     "answer_preds": answer_preds,
        #     "explanation_preds": explanation_preds
        # }

        returned_response = {
            "statement": statement_item,
            "question": question_item,
            "question_type": question_type_item,
            "answer": answer_item,
            "explanation": explanation_item,
            "difficulty": difficulty_item,
            "ability": ability_item,
            "image": image_item,
            "modality": modality_item,
            "source": source_item,
            "answer_preds": answer_preds,
            "explanation_preds": explanation_preds,
            "raw_preds": raw_preds
        }

        field_responses[key] = returned_response

        with open(save_path, 'a', encoding='utf-8') as f:
            json.dump({key: returned_response}, f, ensure_ascii=False, indent=4)
            f.write('\n')

    return field_responses