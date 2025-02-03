import json, os, sys
import requests

from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from typing import List

load_dotenv(dotenv_path=find_dotenv())
api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('MODEL')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

openai = OpenAI()

links_system_prompt = (
    "You are given a list of links from a webpage."
    " Decide which links would be most relevant to include in a brochure about the company."
    " Links like About, Careers, Jobs, or Company pages."
    " Response in JSON as in this example:\n"
)
links_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page": "url": "https://another.full.url/careers"}
    ]
}
"""

links_user_prompt = (
    "Here is a list of links from: {}. Please decide which are relevant for a brochure about the company."
    " Respond with the full URL in JSON format. Do not include Terms of Service, Privacy, or email links."
    " Links (some might be relevant):\n{}"
)

class Website:
    def __init__(self, url):
        self.url = url
        self.text = ''
        self.links = []

        response = requests.get(url, headers=headers)
        self.body = response.content
        
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"

        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()

            self.text = soup.body.get_text(separator="\n", strip=True)
            links = [link.get('href') for link in soup.find_all('a')]
            self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title: {self.title}\nWebpage Contents:\n{self.text}\n\n"

## Get initial information and links
company_name = "HuggingFace"
web = Website("https://huggingface.co")
links_user_prompt = links_user_prompt.format(web.url, "\n".join(web.links))

response = openai.chat.completions.create(
    model = model,
    messages = [
        {"role": "system", "content": links_system_prompt},
        {"role": "user", "content": links_user_prompt}
    ],
    response_format={"type": "json_object"}
)

links_result = json.loads(response.choices[0].message.content)

## Build the brochure
details = "Landing Page:\n"
details += web.get_contents()

for link in links_result['links']:
    details += f"\n\n{link['type']}\n"
    details += Website(link["url"]).get_contents()

system_prompt = (
    "You are an assistant that analyzes the contents of several relevant pages from a company website"
    " and creates a short brochure about the company for prospective customers, investors and recruits."
    " Respond in markdown. Include details of company culture, customers and careers/jobs if you have the information."
)

user_prompt = (
    "You are looking at a company called: {}. Here are the contents of its landing page and other relevant pages."
    " Use this information to build a short brochure of the company in markdown.\n\n{}"
)

user_prompt = user_prompt.format(company_name, details)

response = openai.chat.completions.create(
    model = model,
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
)

brochure = response.choices[0].message.content
brochure = brochure.replace("```","").replace("markdown", "")

with open(f'brochure.md', 'w') as file:
    file.write(brochure)


