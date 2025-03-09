import json, math, os, pickle, random, re, sys

import matplotlib.pyplot as plt
import numpy as np

from anthropic import Anthropic
from collections import Counter
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import login
from items import Item
from openai import OpenAI
from testing import Tester

load_dotenv(dotenv_path=find_dotenv())

openai_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_MODEL')
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
anthropic_model = os.getenv('ANTHROPIC_MODEL')

openai = OpenAI()
claude = Anthropic()

## Load up our data
with open('train.pkl', 'rb') as file:
    train = pickle.load(file)

with open('test.pkl', 'rb') as file:
    test = pickle.load(file)

fine_tune_train = train[:200]
fine_tune_validation = train[200:250]


## Construct prompt for model
def messages_for_train(item):
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": f"Price is ${item.price:.2f}"}
    ]


def make_jsonl(items):
    result = ""

    for item in items:
        messages = messages_for_train(item)
        messages_str = json.dumps(messages)
        result += '{"messages": ' + messages_str +'}\n'

    return result.strip()

def write_jsonl(items, filename):
    with open(filename, "w") as f:
        jsonl = make_jsonl(items)
        f.write(jsonl)

write_jsonl(fine_tune_train, "fine_tune_train.jsonl")
write_jsonl(fine_tune_validation, "fine_tune_validation.jsonl")

with open("fine_tune_train.jsonl", "rb") as f:
    train_file = openai.files.create(file=f, purpose="fine-tune")

with open("fine_tune_validation.jsonl", "rb") as f:
    validation_file = openai.files.create(file=f, purpose="fine-tune")

wandb_integration = {"type": "wandb", "wandb": {"project": "gpt-pricer"}}

## Commented out since we already fine-tuned a model
#openai.fine_tuning.jobs.create(
#    training_file=train_file.id,
#    validation_file=validation_file.id,
#    model="gpt-4o-mini-2024-07-18",
#    seed=42,
#    hyperparameters={"n_epochs": 1},
#    integrations = [wandb_integration],
#    suffix="pricer"
#)

jobs = openai.fine_tuning.jobs.list(limit=1)

print("========== JOBS ==========")
print(jobs)
print()

job_id = openai.fine_tuning.jobs.list(limit=1).data[0].id
status = openai.fine_tuning.jobs.list(limit=1).data[0].status

print(f'{job_id}: {status}')

## Need to wait for fine-tuning to finish
## We did this manually so not to have to code checking/waiting

job = openai.fine_tuning.jobs.retrieve(job_id)

print()
print("========== JOB ==========")
print(job)
print()

job_events = openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10).data

print("========== JOB EVENTS ==========")
print(job_events)
print()

fine_tuned_model_name = job.fine_tuned_model

print("========== FINE-TUNED MODEL NAME ==========")
print(fine_tuned_model_name)
print()

if not fine_tuned_model_name:
	sys.exit()


## Construct prompt for model
def messages_for_test(item):
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

def gpt_fine_tuned(item):
    response = openai.chat.completions.create(
        model=fine_tuned_model_name, 
        messages=messages_for_test(item),
        seed=42,
        max_tokens=7
    )
    reply = response.choices[0].message.content

    return get_price(reply)

Tester.test(gpt_fine_tuned, test)

