# Multimodal EDA Benchmarking

[简体中文CN](README_CN.md)

## Project Overview

The Multimodal EDA Benchmarking project aims to provide a framework for evaluating and comparing the performance of different models on multimodal data. This project supports various data formats and models, helping researchers and developers better understand the capabilities of models in handling multimodal data.

## TODO

### Text-only Model
- [x] GPT-3.5-turbo
- [ ] Llama2-7b-chat-hf
- [ ] Llama2-13b-chat-hf
- [ ] Llama3-8b-instruct
- [ ] mistrial-7b

### Multi-modal Model
- [x] GPT-4o
- [x] MiniGPT4-vicuna-7b
- [ ] MiniGPT4-vicuna-13b
- [x] GPT-4
- [ ] GPT-4-turbo
- [ ] GPT-4-vision



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