import asyncio
import fastapi_poe as fp
import json

# 将字符串解析为JSON
def parse_string_to_json(string):
    try:
        return json.loads(string)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None

# Create an asynchronous function to encapsulate the async for loop
async def get_responses(api_key, messages):
    response = []
    async for partial in fp.get_bot_response(messages=messages, bot_name="GPT-3.5-Turbo", api_key=api_key):
        text = partial.raw_response['text']
        '{"text": " today"}'
        text = parse_string_to_json(text)
        response.append(text['text'])
        
    print(''.join(response))


 
# Replace <api_key> with your actual API key, ensuring it is a string.
api_key = "GwpNvRYkJDExNG7HJLHHpmxuP0yEoncdlsx0bK7dlG0"
'''
Literal["system", "user", "bot"]
'''
message = fp.ProtocolMessage(role="system", content="You are a helpful assistant.")
message = fp.ProtocolMessage(role="user", content="I'm trying to port ORFS to a TSMC process. During the detailed route step, I get a lot of DRT-0073 errors. Utilization is set to 20% at the beginning of the P&R flow and it is a pretty simple design. Give me some tips about what to look into?\nI have tried to run the pin_access command directly but the -verbose parameter does not seem to work as I don't get any more information")

print("message")
# Run the event loop
# For Python 3.7 and newe   r
asyncio.run(get_responses(api_key, [message]))

# For Python 3.6 and older, you would typically do the following:
# loop = asyncio.get_event_loop()
# loop.run_until_complete(get_responses(api_key))
# loop.close()