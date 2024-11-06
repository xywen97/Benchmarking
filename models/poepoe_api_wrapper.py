from http import client
from poe_api_wrapper import AsyncPoeApi, PoeApi
import asyncio

# 提供有效的 p-b 和 p-lat cookies
p_b_cookie = "0zeKwLxBhWbvbWXhASy9uw%3D%3D"
p_lat_cookie = "8efnoyFlqM4vYia%2BQO1Zp1yMDI0%2BdnWT4mOmz%2B7VlA%3D%3D"

tokens = {
    'p-b': p_b_cookie, 
    'p-lat': p_lat_cookie
}

# async def main():
#     client = await AsyncPoeApi(tokens=tokens).create()
#     message = "hello"
#     async for chunk in client.send_message(bot="claude_3_sonnet_200k", message=message):
#         print(chunk["response"], end='', flush=True)
        
# asyncio.run(main())

client = PoeApi(tokens=tokens)

while True:
    history = []
    while True:
        message = input("请输入消息（输入 'exit' 退出）：")
        if message.lower() == 'exit':
            break
        history.append(message)
        for chunk in client.send_message(bot="gpt3_5", message=message):
            print(chunk["response"], end='', flush=True)