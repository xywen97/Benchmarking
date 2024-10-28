import requests
import json
from openai import OpenAI

api_key = "sk-QpHUrsblHgB7kAzcpwLmrFz3yKKTiFVlFOW2vgVc7ARfqsXR"
base_url = "https://a.fe8.cn/v1"

# api_key = "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"
# base_url = "https://api.openai-forward.com/v1/"

def image_captioning(image_path=None):
    url = 'http://127.0.0.1:5010/generate_caption'
    headers = {'Content-Type': 'application/json'}
    
    if image_path is not None:
        data = json.dumps({"image_path": image_path})
        print(f"data: {data}")
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            return "Error: " + str(response.status_code)
    else:
        return "Error: No image path provided"


def query_llama2_7b(prompt, images=None):
    url = 'http://127.0.0.1:5000/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

def query_llama2_13b(prompt, images=None):
    url = 'http://127.0.0.1:5002/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_llama3_8b_instruct(prompt, images=None):
    url = 'http://127.0.0.1:5003/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_mistrial_7b(prompt, images=None):
    url = 'http://127.0.0.1:5001/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)
    
def query_gpt35(prompt, images=None):
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


def query_gpt4v(prompt, images=None):
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

def query_gpt4o(prompt, images=None):
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

    return llm_outputs

def query_gpt4_turbo(prompt, images=None):
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


def query_gpt4(prompt, images=None):
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
