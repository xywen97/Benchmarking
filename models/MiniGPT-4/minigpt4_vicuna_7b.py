import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from typing import Optional

from flask import Flask, request, jsonify

app = Flask(__name__)

# bash script
# python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=1, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

@app.route('/generate', methods=['POST'])
def generate_text():
    # API implementation here
    # raise NotImplementedError

    # Get prompt from the request
    data = request.json
    text = data.get("prompt", "")
    img_paths = data.get("image_paths", [])

    print(f"""
        This is the minigpt4-vicuna7b script.
        The prompt is: {text}
        The image paths are: {img_paths}
    """)

    try:
        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
        chat_state = CONV_VISION.copy()
        img_list = []
        """
        upload multiple images
        """
        for i in range(len(img_paths)):
            print(f"upload image: {i}, {img_paths[i]}")
            chat.upload_img(img_paths[i], chat_state, img_list)

        print("encoding images...")
        chat.encode_img(img_list)

        print("ask models....")
        chat.ask(text, chat_state)

        num_beams = 1
        temperature = 0.7
        print("query models....")
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=num_beams,
                                  temperature=temperature,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
        print(llm_message)

        return jsonify(llm_message)
    except Exception as e:
        return jsonify(str(e))


def query(text: str, img_paths: Optional[str]) -> str:
    # API implementation here
    # raise NotImplementedError
    try:
        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
        chat_state = CONV_VISION.copy()
        img_list = []
        """
        upload multiple images
        """
        for i in range(len(img_paths)):
            chat.upload_img(img_paths[i], chat_state, img_list)

        chat.encode_img(img_list)

        chat.ask(text, chat_state)

        num_beams = 1
        temperature = 0.7
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=num_beams,
                                  temperature=temperature,
                                  max_new_tokens=300,
                                  max_length=2000)[0]

        return jsonify(llm_message)
    except Exception as e:
        return jsonify(str(e))

if __name__ == '__main__':
    # model_name = 'MiniGPT-4'
    text = """Based on the given AD580 specification document, answer the following questions. Calculate the total error in millivolts for the AD580L device given the line regulation is 2 mV/V and the input varies from 5V to 10V. Filling in blanks question. Please answer in Json format and return Json object only. The response format should be: {{"answer": it should be a number/yes/no/not a sentence, "explanation": no more than 3 sentences for explanation on your thought to give the answer.}}"""
    image_paths = ["/home/xiangyu/project/multimodalEDABenchmarking/data/spec/5962_8686101XA_page1.png", "/home/xiangyu/project/multimodalEDABenchmarking/data/spec/5962_8686101XA_page1.png"]

    # print("Response: ", query(text, image_paths))

    app.run(debug=False, host='0.0.0.0', port=5004)
