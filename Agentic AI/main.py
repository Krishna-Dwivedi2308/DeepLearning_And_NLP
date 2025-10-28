from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import requests
import json
client=OpenAI()

def get_weather(city:str):
    url=f"https://wttr.in/{city}?format=%C+%t"
    response=requests.get(url)
    # print(response.text)
    return response.text



query=input('Enter the name of the city: ')
response=client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": f"Give me the weather in {query}"},
    ],
    tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "Name of the city"}
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
)
message = response.choices[0].message
# print(message)
if message.tool_calls:
    tool_call = message.tool_calls[0]
    if tool_call.function.name == "get_weather":
        # Parse the arguments the model decided to use
        
        args = json.loads(tool_call.function.arguments)
        city = args["city"]
        # Actually run the Python function
        weather_result = get_weather(city)
        print(f"Weather in {city}: {weather_result}")
else:
    print(message.content)