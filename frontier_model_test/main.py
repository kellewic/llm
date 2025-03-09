import csv, json, math, os, pickle, random, re, sys

import matplotlib.pyplot as plt
import numpy as np

from anthropic import Anthropic
from collections import Counter
from dotenv import find_dotenv, load_dotenv
from groq import Groq
from huggingface_hub import login
from items import Item
from openai import OpenAI
from testing import Tester

load_dotenv(dotenv_path=find_dotenv())

openai_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_MODEL')
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
anthropic_model = os.getenv('ANTHROPIC_MODEL')
google_key = os.getenv('GOOGLE_API_KEY')
google_model = os.getenv('GOOGLE_MODEL')
groq_key = os.getenv('GROQ_API_KEY')
groq_model = os.getenv('GROQ_MODEL')

openai = OpenAI()
claude = Anthropic()
google = OpenAI(api_key=google_key, base_url=os.getenv('GOOGLE_API_URL'))
groq = Groq(api_key=groq_key)


## Load up our data
with open('train.pkl', 'rb') as file:
    train = pickle.load(file)

with open('test.pkl', 'rb') as file:
    test = pickle.load(file)

## Load human evaluator data
def human_predictions():
	human_predictions = []
	with open('human_output.csv', 'r', encoding="utf-8") as csvfile:
		reader = csv.reader(csvfile)

		for row in reader:
			human_predictions.append(float(row[1]))

	def human_pricer(item):
		idx = test.index(item)
		return human_predictions[idx]

	Tester.test(human_pricer, test)

## Construct prompt for model
def messages_for(item):
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Price is $"}
    ]

## Get price from string in case the models don't reply with just a price
def get_price(s):
    s = s.replace('$','').replace(',','')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0

def run_openai_model(item):
	response = openai.chat.completions.create(
		model=openai_model, 
		messages=messages_for(item),
		seed=42,
		max_completion_tokens=5
	)
	reply = response.choices[0].message.content

	return get_price(reply)

def openai_model_predictions(model):
	global openai_model
	openai_model = model

	Tester(run_openai_model, test, title=openai_model).run()

def run_anthropic_model(item):
    messages = messages_for(item)
    system_message = messages[0]['content']
    messages = messages[1:]
    response = claude.messages.create(
        model=anthropic_model,
        max_tokens=5,
        system=system_message,
        messages=messages
    )
    reply = response.content[0].text

    return get_price(reply)

def anthropic_model_predictions(model):
	global anthropic_model
	anthropic_model = model

	Tester(run_anthropic_model, test, title=anthropic_model).run()

def run_google_model(item):
	response = google.chat.completions.create(
		model=google_model, 
		messages=messages_for(item),
		max_completion_tokens=5
	)
	reply = response.choices[0].message.content

	return get_price(reply)

def google_model_predictions(model):
	global google_model
	google_model = model

	Tester(run_google_model, test, title=google_model).run()


def run_groq_model(item):
	response = groq.chat.completions.create(
		model=groq_model, 
		messages=messages_for(item),
		seed=42,
		max_completion_tokens=5
	)
	reply = response.choices[0].message.content

	return get_price(reply)

def groq_model_predictions(model):
	global groq_model
	groq_model = model

	Tester(run_groq_model, test, title=groq_model).run()

