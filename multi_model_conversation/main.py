import os, sys
import anthropic
import google.generativeai

from dotenv import find_dotenv, load_dotenv
from itertools import chain
from openai import OpenAI

load_dotenv(dotenv_path=find_dotenv())

openai_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_MODEL')

anthropic_key = os.getenv('ANTHROPIC_API_KEY')
anthropic_model = os.getenv('ANTHROPIC_MODEL')

google_key = os.getenv('GOOGLE_API_KEY')
google_model = os.getenv('GOOGLE_MODEL')

openai = OpenAI()
anthropic = anthropic.Anthropic()
google = OpenAI(
    api_key = google_key,
    base_url = os.getenv('GOOGLE_API_URL')
)

openai_system = (
    "You are a chatbot who is very argumentative. You disagree with anything in the"
    " conversation and challenge everything in a snarky way. Your name is Todd. You"
    " will prefix all responses with 'Todd: '."
)

anthropic_system = (
    "You are a polite and courteous chatbot. You try to agree with everything the other"
    " person says or find common ground. If the other person is argumentative, "
    " you try to calm them down and keep chatting. Your name is Brandon. You will "
    " prefix all responses with 'Brandon: '."
)

google_system = (
    "You are a world-renowned couples therapy counselor."
    " You have all the schooling and knowledge of Sigmond Freud."
    " You also have a PhD in Family Therapy."
    " You speak as Sigmond Freud would."
    " Your specialty is analyzing conversations and providing expert feedback."
    " Your name is Dr. Antilla. You will prefix all responses with 'Dr. Antilla: '."
)

gpt_messages = ["Todd: Hi there"]
claude_messages = ["Brandon: Hi"]
google_messages = ['']

def call_openai():
    messages = [{"role": "system", "content": openai_system}]

    for gpt, claude, gemini in zip(gpt_messages, claude_messages, google_messages):
        messages.append({"role": "assistant", "content": gpt})
        messages.append({"role": "user", "content": "{}\n{}".format(claude, gemini)})

    completion = openai.chat.completions.create(
        model = openai_model,
        messages = messages
    )

    return completion.choices[0].message.content


def call_anthropic():
    messages = []

    for gpt, claude, gemini in zip(gpt_messages, claude_messages, google_messages):
        messages.append({"role": "user", "content": "{}\n{}".format(gemini, gpt)})
        messages.append({"role": "assistant", "content": claude})

    messages.append({"role": "user", "content": gpt_messages[-1]})

    message = anthropic.messages.create(
        model = anthropic_model,
        system = anthropic_system,
        messages = messages,
        max_tokens = 500
    )

    return message.content[0].text


def call_google():
    messages = [{"role": "system", "content": google_system}]
    user = ''

    for gpt, claude, gemini in zip(gpt_messages, claude_messages, google_messages):
        if gemini == '':
            user = "{}\n{}\n".format(gpt, claude)

        else:
            messages.append({"role": "user", "content": "{}\n{}".format(gpt, claude)})
            messages.append({"role": "assistant", "content": gemini})

    messages.append({"role": "user", "content": "{}{}\n{}".format(user, gpt_messages[-1], claude_messages[-1])})

    completion = google.chat.completions.create(
        model = google_model,
        messages = messages
    )

    return completion.choices[0].message.content

def output_text(text):
    print(text, end="\n\n")

print()
output_text(gpt_messages[0])
output_text(claude_messages[0])

for i in range(1, 5):
    gpt_next = call_openai()
    output_text(gpt_next)
    gpt_messages.append(gpt_next)

    claude_next = call_anthropic()
    output_text(claude_next)
    claude_messages.append(claude_next)

    google_next = call_google()
    output_text(google_next)
    google_messages.append(google_next)

