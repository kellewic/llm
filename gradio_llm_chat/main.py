import os, sys
import anthropic
import gradio as gr

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=find_dotenv())

openai_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_MODEL')
openai = OpenAI()

system_message = (
    "You are a helpful assistant that responds in Markdown"
)

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    stream = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        stream=True
    )

    result = ''
    for chunk in stream:
        result += chunk.choices[0].delta.content or ''
        yield result


view = gr.ChatInterface(
    fn=chat,
    type="messages"
)
view.launch(share=True)
