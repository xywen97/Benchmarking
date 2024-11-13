# Multimodal EDA Benchmarking

[简体中文CN](README_CN.md)

## Project Overview

The Multimodal EDA Benchmarking project aims to provide a framework for evaluating and comparing the performance of different models on multimodal data. This project supports various data formats and models, helping researchers and developers better understand the capabilities of models in handling multimodal data.

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

### Multi-modal Model
- [x] GPT-4o
- [x] MiniGPT4-vicuna-7b
- [ ] MiniGPT4-vicuna-13b
- [x] GPT-4
- [x] GPT-4-turbo
- [x] GPT-4-vision
- [x] Reka series
- [x] Blip + T5
- [x] Llama3.2-11b-vision-instruct
- [x] Llama3.2-90b-vision-instruct
- [x] Kosmos2
- [x] QWen VL
- [x] QWen VL Plus
- [x] QWen VL Max
- [ ] Bunny 1.0 3b (some bugs on the same device requirement)
- [ ] hpt 1.5 edge (some bugs about input of size)
- [x] internlm xcomposer vl 7b

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