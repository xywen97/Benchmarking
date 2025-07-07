# vLLM Demo

This directory contains scripts and resources for testing and demonstrating the capabilities of vLLM, specifically focusing on Qwen models.

## Contents

- `vLLM_example.py`: A basic example script demonstrating how to use vLLM for model inference.
- `adaptive_testing_script.py`: An adaptive script that attempts to load and run Qwen models using vLLM if available, or falls back to standard generation methods.
- `vision_language_testing_script.py`: A comprehensive script for testing vision-language models, supporting various configurations and datasets.
- `dataloader.py`: Utility functions for loading and processing data for model testing.

## Usage

1. **vLLM Example**
   - Run `vLLM_example.py` to see a basic example of using vLLM for model inference.

2. **Adaptive Testing**
   - Use `adaptive_testing_script.py` to automatically test Qwen models. The script will attempt to use vLLM if available, otherwise it will use standard methods.

3. **Vision-Language Testing**
   - Execute `vision_language_testing_script.py` for a detailed evaluation of vision-language models. This script supports various datasets and configurations.

## Requirements

- Python 3.10
- PyTorch
- Transformers
- vLLM

## Notes

- The `dataloader.py` script provides utility functions for loading data, which are used across different testing scripts.
- Ensure that the CUDA device is set correctly in the scripts if using GPU acceleration.

For more detailed information, refer to the comments within each script.