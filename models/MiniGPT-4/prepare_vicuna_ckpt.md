## How to Prepare Vicuna Weight
Vicuna is an open-source LLAMA-based LLM that has a performance close to ChatGPT. 
We currently use the v0 version of Vicuna-13B. 

To prepare Vicuna’s weight, first download Vicuna’s **delta** weight from [https://huggingface.co/lmsys/vicuna-13b-delta-v0](https://huggingface.co/lmsys/vicuna-13b-delta-v0). 
In case you have git-lfs installed (https://git-lfs.com), this can be done by

```
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v0  # more powerful, need at least 24G gpu memory
# or
git clone https://huggingface.co/lmsys/vicuna-7b-delta-v0  # smaller, need 12G gpu memory
```

Note that this is not directly the working weight, but the difference between the working weight and the original weight of LLAMA-13B. (Due to LLAMA’s rules, we cannot distribute the weight of LLAMA.)

Then, you need to obtain the original LLAMA-7B or LLAMA-13B weights in the HuggingFace format 
either following the instruction provided by HuggingFace 
[here](https://huggingface.co/docs/transformers/main/model_doc/llama) or from the Internet. 

When these two weights are ready, we can use tools from Vicuna’s team to create the real working weight.
First, Install their library that is compatible with v0 Vicuna by

```
pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
```

Then, run the following command to create the final working weight

```
python -m fastchat.model.apply_delta --base /path/to/llama-13bOR7b-hf/  --target /path/to/save/working/vicuna/weight/  --delta /path/to/vicuna-13bOR7b-delta-v0/
```

Now you are good to go!

### NOTE

If you meet the problem of: 
```
Traceback (most recent call last):
  File "/home/xiangyu/.conda/envs/benchmark/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/xiangyu/.conda/envs/benchmark/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/xiangyu/.conda/envs/benchmark/lib/python3.10/site-packages/fastchat/model/apply_delta.py", line 9, in <module>
    import torch
  File "/home/xiangyu/.conda/envs/benchmark/lib/python3.10/site-packages/torch/__init__.py", line 368, in <module>
    from torch._C import *  # noqa: F403
ImportError: /home/xiangyu/.conda/envs/benchmark/lib/python3.10/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefi
ned symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
```

You can export in the ./.bashrc file

```
export LD_LIBRARY_PATH=/home/xiangyu/.conda/envs/benchmark/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```