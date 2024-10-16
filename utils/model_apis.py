import requests
import json
from openai import OpenAI

# api_key = "sk-89576ilSZoBXCFmGlJPxBEzpYGDRe17MBudXmnQmhMxmaC3x"
# base_url = "https://a.fe8.cn/v1/"

api_key = "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"
base_url = "https://api.openai-forward.com/v1/"

def query_llama2_7b(prompt):
    url = 'http://127.0.0.1:5000/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_llama2_13b(prompt):
    url = 'http://127.0.0.1:5002/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_llama3_8b_instruct(prompt):
    url = 'http://127.0.0.1:5003/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_mistrial_7b(prompt):
    url = 'http://127.0.0.1:5001/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_gpt35(prompt):
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
        temperature=0.0,
        stream=False  
    )
    llm_outputs = llm_response.choices[0].message.content
    return llm_outputs

def query_gpt4o(prompt):
    openai_api_key = api_key
    openai_api_base = base_url
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
    return llm_outputs

def query_gpt4(prompt):
    openai_api_key = api_key
    openai_api_base = base_url
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
