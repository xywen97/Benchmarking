import requests
import json

def send_prompt_to_server(prompt):
    # llama_2_7b_chat_hf
    # url = 'http://127.0.0.1:5000/generate'

    # mistrial_7b_0.1
    # url = 'http://127.0.0.1:5001/generate'

    # llama_2_13b_chat_hf
    url = 'http://127.0.0.1:5002/generate'

    # llama_3_8b_instruct
    # url = 'http://127.0.0.1:5003/generate'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"prompt": prompt})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return "Error: " + str(response.status_code)

if __name__ == "__main__":
    prompt = "The point $(0,0)$ is reflected across the vertical line $x = 1$. Its image is then reflected across the line $y=2$. What are the coordinates of the resulting point?"
    generated_text = send_prompt_to_server(prompt)
    print("Generated Text:", generated_text)
