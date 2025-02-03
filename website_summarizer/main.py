import os, sys
import ollama, requests

from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=find_dotenv())
api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('OPENAI_MODEL')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

openai = OpenAI()

system_prompt = (
    "You're an assistant that analyzes website contents."
    " You provide a short summary, ignoring navigation-related text."
    " Respond in Markdown."
)

user_prompt = (
    "Here's web page text with the URL and Title embedded at the beginning."
    " The page content follows the Title."
    " Provide a short summary of this web page in Markdown."
    " If it includes news or announcements, summarize those too.\n\n"
)

def get_webpage_content(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.string if soup.title else "No title found"

    for irrelevant in soup.body(["script", "style", "img", "input"]):
        irrelevant.decompose()

    text = soup.body.get_text(separator="\n", strip=True)

    return f"{url=}\n{title=}\n\n{text}"

user_prompt += get_webpage_content("https://www.gradio.app/guides/quickstart")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

## Get response from OpenAI
response = openai.chat.completions.create(
    model = model,
    messages = messages
)

with open(f'{model}-response.md', 'w') as file:
    file.write(response.choices[0].message.content)

## Get response from local model
model = os.getenv('LOCAL_MODEL')

response = ollama.chat(
    model=model,
    messages=messages
)

with open(f'{model}-response.md', 'w') as file:
    file.write(response['message']['content'])

