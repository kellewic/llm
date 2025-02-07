import json, os, sys
import anthropic
import gradio as gr

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=find_dotenv())

openai_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_MODEL')
openai = OpenAI()

system_message = (
    "You are a helpful assistant for an airline called FlightAI."
    " Give short, courteous answers, no more than 1 sentence."
    " Always be accurate. If you don't know the answer, say so."
)

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

price_function = {
    "name": "get_ticket_price",
    "description": "Get price of return ticket to destination city. Call when you need ticket prices, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": price_function}]

def get_ticket_price(destination_city='Unknown'):
    return {'price': ticket_prices.get(destination_city.lower(), "Unknown")}


def handle_tool_call(message):
    response = []

    for tool_call in message.tool_calls:
        function = tool_call.function
        name = function.name

        arguments = json.loads(function.arguments)
        arguments.update(globals()[name](**arguments))

        response.append({
            "role": "tool",
            "content": json.dumps(arguments),
            "tool_call_id": tool_call.id
        })

    return response


def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=openai_model, messages=messages, tools=tools)

    if response.choices[0].finish_reason == 'tool_calls':
        message = response.choices[0].message
        messages.append(message)

        response = handle_tool_call(message)
        messages.extend(response)

        response = openai.chat.completions.create(model=openai_model, messages=messages)

    return response.choices[0].message.content

view = gr.ChatInterface(fn=chat, type="messages").launch(share=True)

