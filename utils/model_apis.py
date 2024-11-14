from pyexpat.errors import messages
import requests
import json
from openai import OpenAI
import fastapi_poe as fp
import asyncio
from reka.client import Reka
from reka import ChatMessage

api_key = "sk-QpHUrsblHgB7kAzcpwLmrFz3yKKTiFVlFOW2vgVc7ARfqsXR"
api_key_poe = "GwpNvRYkJDExNG7HJLHHpmxuP0yEoncdlsx0bK7dlG0"
api_key_reka = "eb696dc825f2cba426aac1b5758ffd4d7d454cfbdd82a9c147d81bfcfdeead3a"
base_url = "https://a.fe8.cn/v1"

# api_key = "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"
# base_url = "https://api.openai-forward.com/v1/"

def image_captioning(image_path=None):
    url = 'http://127.0.0.1:5010/generate_caption'
    headers = {'Content-Type': 'application/json'}
    
    if image_path is not None:
        data = json.dumps({"image_path": image_path})
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            return "Error: " + str(response.status_code)
    else:
        return "Error: No image path provided"


def query_reka_series(prompt, images=None, image_captions=None, model_name=None):
    '''
    support image url only
    '''
    model_to_name_mapping = {
        "reka_core": "reka-core",
        "reka_flash": "reka-flash",
        "reka_edge": "reka-edge"
    }

    chosen_model = model_to_name_mapping[model_name]

    client = Reka(api_key=api_key_reka)


    user_contents = [
        {"type": "text", "text": prompt}
    ]
    for image in images:
        user_contents.append(
            {"type": "image_url", "image_url": image}
        )

    messages = [
        ChatMessage(
            content = user_contents,
            role="user"
        )
    ]

    response = client.chat.create(
        messages = messages,
        model = chosen_model,
        max_tokens = 1024,
        temperature = 0.7
    )
    return response.responses[0].message.content


def query_gemini_series(prompt, images=None, image_captions=None, model_name=None):
    model_to_bot_mapping = {
        "gemini_1.0_pro": "Gemini-1.0-pro",
        "gemini_1.5_pro": "Gemini-1.5-pro",
        "gemini_1.5_flash": "Gemini-1.5-Flash"
    }
    bot_name = model_to_bot_mapping[model_name]
    def parse_string_to_json(string):
        try:
            return json.loads(string)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return None
    async def get_responses(api_key_poe, messages):
        response = []
        async for partial in fp.get_bot_response(messages=messages, bot_name=bot_name, api_key=api_key_poe):
            text = partial.raw_response['text']
            text = parse_string_to_json(text)
            response.append(text['text'])
            
        # print(''.join(response))
        return_result = ''.join(response)
        return return_result
    
    message_system = fp.ProtocolMessage(role="system", content="You are a helpful assistant.")
    message_user = fp.ProtocolMessage(role="user", content=prompt)

    result = asyncio.run(get_responses(api_key_poe, [message_system, message_user]))
    return result

def query_claude_series(prompt, images=None, image_captions=None, model_name=None):
    model_to_bot_mapping = {
        "claude_3_sonnet": "Claude-3-Sonnet",
        "claude_3_haiku": "Claude-3-Haiku",
        "claude_3_opus": "Claude-3-Opus"
    }
    bot_name = model_to_bot_mapping[model_name]
    def parse_string_to_json(string):
        try:
            return json.loads(string)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return None
    async def get_responses(api_key_poe, messages):
        response = []
        async for partial in fp.get_bot_response(messages=messages, bot_name=bot_name, api_key=api_key_poe):
            text = partial.raw_response['text']
            text = parse_string_to_json(text)
            response.append(text['text'])
            
        # print(''.join(response))
        return_result = ''.join(response)
        return return_result
    
    message_system = fp.ProtocolMessage(role="system", content="You are a helpful assistant.")
    message_user = fp.ProtocolMessage(role="user", content=prompt)

    result = asyncio.run(get_responses(api_key_poe, [message_system, message_user]))
    return result


def query_llama2_7b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5000/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_mistrial_7b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5001/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_llama2_13b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5002/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_llama3_8b_instruct(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5003/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_minigpt4_vicuna7b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5004/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_instructblip_flan_t5_xl(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5005/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_instructblip_flan_t5_xxl(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5006/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)


def query_blip2_flan_t5_xl(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5007/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_blip2_flan_t5_xxl(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5008/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_chatglm3_6b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5009/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_llama3_1_70b_instruct(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5011/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_llama3_1_8b_instruct(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5012/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_llama3_2_11b_vision_instruct(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5013/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_llama3_2_90b_vision_instruct(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5014/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_kosmos_2_patch14_224(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5015/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)


def query_qwen_2_5_7b_instruct(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5016/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_qwen_2_5_72b_instruct(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5017/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_qwen_2_7b_instruct(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5018/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_qwen_2_72b_instruct(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5019/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_qwen_2_0_5b_instruct(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5020/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    

def query_qwen_vl(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5021/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_qwen_vl_plus(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5022/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_qwen_vl_max(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5023/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_qwen_vl_chat(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5024/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_bunny_1_0_3b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5025/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_fuyu_8b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5026/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_hpt_1_5_edge(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5027/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_internlm_chat_7b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5028/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_internlm_chat_20b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5029/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_internlm_xcomposer_vl_7b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5030/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_internlm_xcomposer2_vl_7b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5031/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_internlm2_chat_7b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5032/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_internlm2_chat_20b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5033/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_internlm2_5_chat_7b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5034/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_internlm2_5_chat_20b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5035/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_miniCPM_V(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5036/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_miniCPM_V2(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5037/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_miniCPM3_4b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5038/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_miniCPM1b_sft_bf16(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5039/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_miniCPM2b_sft_bf16(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5040/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_miniCPM_llama3_v_2_5(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5041/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_internvl2_8b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5042/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_internvl2_40b(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5043/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_internvl_chat_v1_5(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5044/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_internvl_mini_chat_2b_v1_5(prompt, images=None, image_captions=None):
    url = 'http://127.0.0.1:5045/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt, "image_paths": images})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_gpt35(prompt, images=None, image_captions=None):
    openai_api_key = api_key
    openai_api_base = base_url
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    llm_response = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        max_tokens=1024,
        temperature=0.7,
        stream=False  
    )
    llm_outputs = llm_response.choices[0].message.content
    return llm_outputs


def query_gpt4v(prompt, images=None, image_captions=None):
    print("We are using gpt 4v")
    openai_api_key = api_key
    openai_api_base = base_url

    if images:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        # too slow
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": f"You are a helpful assistant. {prompt}. You can refer to the provided images."
                        },
                        {
                            "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images[0]}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "stream": False  
        }

        for image in images:
            payload["messages"][0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            })

        llm_outputs = requests.post("https://a.fe8.cn/v1/chat/completions", headers=headers, json=payload)
        llm_outputs = llm_outputs.json()
        llm_outputs = llm_outputs['choices'][0]['message']['content']

    else:
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        llm_response = client.chat.completions.create(
            messages=messages,
            model="gpt-4-vision-preview",
            max_tokens=1024,
            temperature=0.7,
            stream=False  
        )
        llm_outputs = llm_response.choices[0].message.content

    return llm_outputs

def create_payload(images, prompt: str, model="gpt-4-vision-preview", max_tokens=1024, detail="high"):
    """Creates the payload for the API request."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"You are a helpful assistant. {prompt} You can refer to the provided images.",
                },
            ],
        },
    ]

    for image in images:
        base64_image = image
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": base64_image,
                "detail": detail,
            }
        })

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }

def query_gpt4o(prompt, images=None, image_captions=None):
    openai_api_key = api_key
    openai_api_base = base_url

    if images:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        # too slow
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": f"You are a helpful assistant. {prompt}. You can refer to the provided images."
                        },
                        {
                            "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images[0]}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "stream": False  
        }

        for image in images:
            payload["messages"][0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            })

        llm_outputs = requests.post("https://a.fe8.cn/v1/chat/completions", headers=headers, json=payload)
        llm_outputs = llm_outputs.json()
        llm_outputs = llm_outputs['choices'][0]['message']['content']

    else:
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        llm_response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            max_tokens=1024,
            temperature=0.7,
            stream=False  
        )
        llm_outputs = llm_response.choices[0].message.content

    print(llm_outputs)
    return llm_outputs

def query_gpt4_turbo(prompt, images=None, image_captions=None):
    print("we are using GPT-4_turbo")

    openai_api_key = api_key
    openai_api_base = base_url

    if images:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        # too slow
        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": f"You are a helpful assistant. {prompt}. You can refer to the provided images."
                        },
                        {
                            "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images[0]}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "stream": False  
        }

        for image in images:
            payload["messages"][0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            })

        llm_outputs = requests.post(f"{openai_api_base}/chat/completions", headers=headers, json=payload)
        llm_outputs = llm_outputs.json()
        llm_outputs = llm_outputs['choices'][0]['message']['content']

    else:
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        llm_response = client.chat.completions.create(
            messages=messages,
            model="gpt-4-turbo",
            max_tokens=1024,
            temperature=0.7,
            stream=False  
        )
        llm_outputs = llm_response.choices[0].message.content

    return llm_outputs


def query_gpt4(prompt, images=None, image_captions=None):
    print("we are using GPT-4")

    openai_api_key = api_key
    openai_api_base = base_url

    if images:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        # too slow
        payload = {
            "model": "gpt-4",
            "messages": [
                {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": f"You are a helpful assistant. {prompt}. You can refer to the provided images."
                        },
                        {
                            "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images[0]}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "stream": False  
        }

        for image in images:
            payload["messages"][0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            })

        llm_outputs = requests.post(f"{openai_api_base}/chat/completions", headers=headers, json=payload)
        llm_outputs = llm_outputs.json()
        llm_outputs = llm_outputs['choices'][0]['message']['content']

    else:
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        llm_response = client.chat.completions.create(
            messages=messages,
            model="gpt-4",
            max_tokens=1024,
            temperature=0.7,
            stream=False  
        )
        llm_outputs = llm_response.choices[0].message.content

    return llm_outputs



if __name__ == "__main__":
    prompt = "Compute the area of the circle that passes through all the intersection points of $4x^2 + 11y^2 = 29$ and $x^2 - 6y^2 = 6.$"
    generated_text = query_gpt4(prompt)
    print("Generated Text:", generated_text)
