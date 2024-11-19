# Multimodal EDA Benchmarking

[简体中文CN](README_CN.md)

## Project Overview

The Multimodal EDA Benchmarking project aims to provide a framework for evaluating and comparing the performance of different models on multimodal data. This project supports various data formats and models, helping researchers and developers better understand the capabilities of models in handling multimodal data.

## Tested Models

#### 2024.11.19
- [x] reka_flash
- [x] reka_edge
- [x] blip2_flan_t5_xl
- [x] blip2_flan_t5_xxl
- [x] instructblip_flan_t5_xl
- [x] kosmos2_patch14_224
- [x] minicpm2b_sft_bf16
- [x] minicpm3_4b
- [x] minicpm_llama3_v_2_5
- [x] chatglm3_6b
- [x] gpt4
- [x] yi_chat_6b
- [x] yi_vl_34b

#### 2024.11.18
- [x] llama3_8b_instruct
- [x] llama3_1_70b_instruct (no general)
- [x] llama3_2_11b_vision_instruct
- [x] intern_chat_20b
- [x] internvl_mini_chat_2b_v1_5 (no general)
- [x] internvl2_40b
- [x] internlm_xcomposer_vl_7b
- [x] qwen_2_7b_instruct
- [x] gpt4o
- [x] gpt4_turbo
- [x] gpt4v

#### 2024.11.17
- [x] MiniGPT4-vicuna7b (no general)
- [x] Llama 2-7b
- [x] Llama 2-13b
- [x] Internlm chat 7b
- [x] InternVL-Chat-V1.5
- [x] InternLM2 chat 7b
- [x] InternLM2 Chat 20b
- [x] InternLM2.5 Chat 7b
- [x] InternLM2.5 Chat 20b
- [x] Qwen2 72b instruct
- [x] QWen 2.5 7b instruct
- [x] QWen 2.5 72b instruct
- [x] MiniCPM-V


#### 2024.11.15
- [x] GPT-3.5-turbo
- [x] QWen2 0.5b Instruct
- [x] internvl2-8b
- [x] instructBlip flan T5 xxl
- [x] Llama3.1-8b-instruct
- [x] miniCPM-V2
- [x] yi vl 6b

```
# python main.py --field rtl --model reka_core
# python main.py --field general --model reka_flash (Done)
# python main.py --field general --model reka_edge (Done)
# python main.py --field general --model blip2_flan_t5_xl (Done)
# python main.py --field general --model blip2_flan_t5_xxl (Done)
# python main.py --field rtl --model gemini_1_0_pro
# python main.py --field general --model instructblip_flan_t5_xl (Done)
# python main.py --field general --model kosmos2_patch14_224 (Done)
# python main.py --field general --model minicpm1b_sft_bf16
# python main.py --field general --model minicpm2b_sft_bf16 (Done)
# python main.py --field general --model minicpm3_4b (Done)
# python main.py --field general --model minicpm_llama3_v_2_5 (Done)
# python main.py --field general --model chatglm3_6b (Done)
# python main.py --field general --model gpt4 (Done)
# python main.py --field general --model yi_chat_6b (Done)
# python main.py --field general --model yi_vl_34b (Done)


# 2024.11.18
# python main.py --field general --model llama2_7b (Done)
# python main.py --field general --model llama2_13b (Done)
# python main.py --field rtl --model llama3_1_70b_instruct
# python main.py --field general --model llama3_8b_instruct (Done)
# python main.py --field general --model llama3_2_11b_vision_instruct (Done)
# python main.py --field general --model llama3_2_90b_vision_instruct
# python main.py --field general --model intern_chat_20b (Done)
# python main.py --field rtl --model internvl_mini_chat_2b_v1_5
# python main.py --field general --model internvl2_40b
# python main.py --field general --model internlm_xcomposer_vl_7b (Done)
# python main.py --field rtl --model internlm_xcomposer2_vl_7b
# python main.py --field general --model qwen_2_7b_instruct (Done)
# python main.py --field general --model qwen_vl_chat (Done)
# python main.py --field general --model qwen_vl (Done)
# python main.py --field general --model gpt4o (Done)
# python main.py --field general --model gpt4_turbo (Done)
# python main.py --field general --model gpt4v (Done)
# python main.py --field general --model qwen_2_72b_instruct

# 2024.11.16
# python main.py --field general --model gpt35 (Done)
# python main.py --field general --model qwen_2_0_5b_instruct (Done)
# python main.py --field general --model internvl2_8b (Done)
# python main.py --field rtl --model internvl_chat_v1_5
# python main.py --field general --model instructblip_flan_t5_xxl (Done)
# python main.py --field general --model llama3_1_8b_instruct (Done)
# python main.py --field general --model minicpm_v (Done)
# python main.py --field general --model minicpm_v2 (Done)
# python main.py --field general --model yi_vl_6b (Done)
```


## TODO
### Image Captioning
- [x] Blip-image-captioning-large

### Text-only Model
- [x] GPT-3.5-turbo
- [x] Llama2-7b-chat-hf
- [x] Llama2-13b-chat-hf
- [x] Llama3-8b-instruct
- [x] Llama3.1-8b-instruct
- [x] Llama3.1-70b-instruct
- [ ] mistrial-7b
- [x] ChatGLM3 6b
- [x] QWen2.5 7b Instruct
- [x] QWen2.5 72b Instruct
- [x] QWen2 7b Instruct
- [x] QWen2 72b Instruct
- [x] QWen2 0.5b Instruct
- [x] Internlm chat 7b
- [x] Internlm chat 20b
- [x] Internlm2 chat 7b
- [x] Internlm2 chat 20b
- [x] Internlm2.5 7b chat
- [x] Internlm2.5 20b chat
- [x] miniCPM3-4b
- [x] miniCPM-1b-sft-bf16
- [x] miniCPM-2b-sft-bf16

### Multi-modal Model
- [x] GPT-4o
- [x] MiniGPT4-vicuna-7b
- [ ] MiniGPT4-vicuna-13b
- [x] GPT-4
- [x] GPT-4-turbo
- [x] GPT-4-vision
- [x] Reka series
- [x] instructBlip flan T5 xl
- [x] instructBlip flan T5 xxl
- [x] blip2 flan t5 xl
- [x] blip2 flan t5 xxl
- [x] Llama3.2-11b-vision-instruct
- [x] Llama3.2-90b-vision-instruct
- [x] Kosmos2
- [x] QWen VL
- [x] QWen VL Plus
- [x] QWen VL Max
- [ ] Bunny 1.0 3b (some bugs on the same device requirement)
- [ ] hpt 1.5 edge (some bugs about input of size)
- [x] internlm xcomposer vl 7b
- [x] miniCPM-V
- [x] miniCPM-V2
- [x] miniCPM-LLaMa3-V-2.5
- [x] internvl2-8b
- [x] internvl2-40b
- [x] internvl-chat-v1.5
- [x] mini-internvl-chat-2b-v1.5
- [x] yi vl 6b
- [x] yi vl 34b

### Multi-modal Model (image not ready)
- [x] Gemini series
- [x] Claude series



## Data Format

The data format in the project includes a JSON file and an image directory. The data for each subdomain follows the structure below:

- Image Directory: Such as `general/`, containing image files.
- JSON File: Such as `general.json`, containing information like questions and answers.

An example structure of the JSON file is as follows:
```json
{
    "item_id": {
        "statement": "Initial statement of the question",
        "image": ["List of image names"],
        "question": ["List of questions"],
        "question_type": ["Type of questions"],
        "answer": ["Answers to the questions"],
        "explanation": ["Explanation of the answers"],
        "modality": ["Data modality of the questions"],
        "difficulty": ["Difficulty level of the questions"],
        "ability": ["Model's testing ability on the questions"],
        "source": "Source of the material"
    }
}
```

## Supported Subdomains

- General Knowledge
- Spec
- RTL
- Netlist
- Architecture
- Backend

## Usage Instructions

### Environment Setup

Ensure Python 3.10 is installed, along with the necessary dependencies for the project.

### Running Example

An example is provided in `main.py`, demonstrating how to load data and benchmark models:
python:multimodalEDABenchmarking/main.py
startLine: 4
endLine: 33

### Data Loading

The data loading functionality is implemented in `utils/dataloader.py`, supporting both single and batch data loading.

### Benchmarking

The benchmarking functionality is implemented in `utils/parser.py`, supporting testing of both single and batch questions.

python:multimodalEDABenchmarking/utils/parser.py
startLine: 50
endLine: 140

## Contribution

Contributions to this project are welcome. Please submit a Pull Request or Issue to help us improve.

## License

This project is licensed under the MIT License.