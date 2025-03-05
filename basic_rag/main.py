import glob, os, sys
import gradio as gr

from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=find_dotenv())

openai_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_MODEL')
openai = OpenAI()

system_message = (
    "You are an expert in answering accurate questions about Insurellm, the Insurance Tech company."
    " Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up "
    " if you haven't been provided with relevant context."
)

context = {}

## Get knowledge from files for basic RAG
def get_knowledge(knowledge_type):
    files = glob.glob(f"knowledge-base/{knowledge_type}/*")

    for file in files:
        name = Path(file).stem.split(' ')[-1]
        doc = ''

        with open(file,'r', encoding="utf-8") as f:
            doc = f.read()

        context[name] = doc

## Fill up context with knowledge
get_knowledge("employees")
get_knowledge("products")

def get_relevant_context(message):
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title.lower() in message.lower():
            relevant_context.append(context_details)

    return relevant_context

## Add relevant context to chat
def add_context(message):
    relevant_context = get_relevant_context(message)

    if relevant_context:
        message += "\n\nThe following additional context might be relevant in answering this question:\n\n"

        for relevant in relevant_context:
            message += relevant + "\n\n"

    return message


def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": add_context(message)}]
    stream = openai.chat.completions.create(model=openai_model, messages=messages, stream=True)

    response = ''
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

view = gr.ChatInterface(fn=chat, type="messages").launch(share=True)

