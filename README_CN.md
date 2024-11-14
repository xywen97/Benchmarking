# Multimodal EDA Benchmarking

## 项目概述

Multimodal EDA Benchmarking项目旨在提供一个框架，用于评估和比较不同模型在多模态数据上的表现。该项目支持多种数据格式和模型，帮助研究人员和开发者更好地理解模型在处理多模态数据时的能力。

## TODO

### 图片标题生成
- [x] Blip-image-captioning-large

### 文本模态模型
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

### 多模态模型
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
- [x] miniCPM-V
- [x] miniCPM-V2
- [x] miniCPM-LLaMa3-V-2.5
- [x] internvl2-8b
- [x] internvl2-40b
- [x] internvl-chat-v1.5
- [x] mini-internvl-chat-2b-v1.5

### 多模态模型 (图片端还未成功)
- [x] Gemini series
- [x] Claude series


## 数据格式

项目中的数据格式包括一个JSON文件和一个图像目录。每个子领域的数据都遵循以下结构：

- 图像目录：如`general/`，包含图像文件。
- JSON文件：如`general.json`，包含问题和答案等信息。

JSON文件的结构示例如下：
```json
{
    "item_id": {
        "statement": "问题的初步陈述",
        "image": ["图像名称列表"],
        "question": ["问题列表"],
        "question_type": ["问题类型"],
        "answer": ["问题答案"],
        "explanation": ["答案解释"],
        "modality": ["问题的数据模态"],
        "difficulty": ["问题的难度等级"],
        "ability": ["模型在问题上的测试能力"],
        "source": "材料来源"
    }
}
```

## 支持的子领域

- General Knowledge
- Spec
- RTL
- Netlist
- Architecture
- Backend

## 使用说明

### 环境设置

确保安装了Python 3.10，并安装了项目所需的依赖库。

### 运行示例

在`main.py`中提供了一个示例，展示了如何加载数据并对模型进行基准测试：
python:multimodalEDABenchmarking/main.py
startLine: 4
endLine: 33


### 数据加载

数据加载功能在`utils/dataloader.py`中实现，支持单个和批量数据的加载。

### 基准测试

基准测试功能在`utils/parser.py`中实现，支持对单个问题和批量问题进行测试。

python:multimodalEDABenchmarking/utils/parser.py
startLine: 50
endLine: 140


## 贡献

欢迎对本项目进行贡献。请提交Pull Request或Issue以帮助我们改进。

## 许可证

本项目采用MIT许可证。
