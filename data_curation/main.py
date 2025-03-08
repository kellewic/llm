import os, pickle, random, sys

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from datasets import load_dataset, Dataset, DatasetDict
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import login

from items import Item
from loaders import ItemLoader

load_dotenv(dotenv_path=find_dotenv())

hf_key = os.getenv('HF_API_KEY')
hf_user = os.getenv('HF_USER')

login(hf_key)
Item.init_tokenizer()

items = []

## Load data, clean and tokenize it
def main():
	global items

	dataset_names = [
		"Automotive",
		"Electronics",
		"Office_Products",
		"Tools_and_Home_Improvement",
		"Cell_Phones_and_Accessories",
		"Toys_and_Games",
		"Appliances",
		"Musical_Instruments",
	]

	for dataset_name in dataset_names:
		loader = ItemLoader(dataset_name)
		items.extend(loader.load(workers=6))

## For the benefit of ProcessPoolExecutor
if __name__ == '__main__':
	main()

print(f"Items: {len(items):,} items")

## Plot token size
tokens = [item.token_count for item in items]
plt.figure(figsize=(15, 6))
plt.title(f"Token counts: Avg {sum(tokens)/len(tokens):,.1f} and highest {max(tokens):,}\n")
plt.xlabel('Length (tokens)')
plt.ylabel('Count')
plt.hist(tokens, rwidth=0.7, color="skyblue", bins=range(0, 300, 10))
plt.savefig('tokens.png')
plt.close()

## Plot price distribution
def plot_prices(items, image_name):
	prices = [item.price for item in items]
	plt.figure(figsize=(15, 6))
	plt.title(f"Prices: Avg {sum(prices)/len(prices):,.1f} and highest {max(prices):,}\n")
	plt.xlabel('Price ($)')
	plt.ylabel('Count')
	plt.hist(prices, rwidth=0.7, color="blueviolet", bins=range(0, 1000, 10))
	plt.savefig(f'{image_name}.png')
	plt.close()

## Plot category counts
def plot_category_counts(items, image_name):
	category_counts = Counter()
	for item in items:
		category_counts[item.category]+=1

	categories = category_counts.keys()
	counts = [category_counts[category] for category in categories]

	# Bar chart by category
	plt.figure(figsize=(15, 6))
	plt.bar(categories, counts, color="goldenrod")
	plt.title('How many in each category')
	plt.xlabel('Categories')
	plt.ylabel('Count')

	plt.xticks(rotation=30, ha='right')

	# Add value labels on top of each bar
	for i, v in enumerate(counts):
		plt.text(i, v, f"{v:,}", ha='center', va='bottom')

	plt.savefig(f'{image_name}.png')
	plt.close()

plot_prices(items, 'prices')
plot_category_counts(items, 'categories')

## Make dataset prices and categories more balanced
slots = defaultdict(list)
for item in items:
    slots[round(item.price)].append(item)

np.random.seed(42)
random.seed(42)
sample = []
for i in range(1, 1000):
    slot = slots[i]
    if i>=240:
        sample.extend(slot)
    elif len(slot) <= 1200:
        sample.extend(slot)
    else:
        weights = np.array([1 if item.category=='Automotive' else 5 for item in slot])
        weights = weights / np.sum(weights)
        selected_indices = np.random.choice(len(slot), size=1200, replace=False, p=weights)
        selected = [slot[i] for i in selected_indices]
        sample.extend(selected)

print(f"There are {len(sample):,} items in the sample")

## Plot new price and category distributions
plot_prices(sample, 'prices_new')
plot_category_counts(sample, 'categories_new')

# How does the price vary with character count of the prompt?
sizes = [len(item.prompt) for item in sample]
prices = [item.price for item in sample]

# Create the scatter plot
plt.figure(figsize=(15, 8))
plt.scatter(sizes, prices, s=0.2, color="red")

# Add labels and title
plt.xlabel('Size')
plt.ylabel('Price')
plt.title('Is there a simple correlation?')
plt.savefig('price_vs_characters.png')
plt.close()

def report(item):
    prompt = item.prompt
    tokens = Item.tokenizer.encode(item.prompt)
    print(prompt)
    print(tokens[-10:])
    print(Item.tokenizer.batch_decode(tokens[-10:]))

report(sample[100])

random.seed(42)
random.shuffle(sample)
train = sample[:400_000]
test = sample[400_000:402_000]
print(f"Divided into a training set of {len(train):,} items and test set of {len(test):,} items")

print(train[0].prompt)
print(test[0].test_prompt())

# Plot the distribution of prices in the first 250 test points
prices = [float(item.price) for item in test[:250]]
plt.figure(figsize=(15, 6))
plt.title(f"Avg {sum(prices)/len(prices):.2f} and highest {max(prices):,.2f}\n")
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.hist(prices, rwidth=0.7, color="darkblue", bins=range(0, 1000, 10))
plt.savefig('prices_test.png')
plt.close()

## Upload dataset to HF
train_prompts = [item.prompt for item in train]
train_prices = [item.price for item in train]
test_prompts = [item.test_prompt() for item in test]
test_prices = [item.price for item in test]

# Create a Dataset from the lists
train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

dataset.push_to_hub(f'{hf_user}/pricer-data', private=True)

## pickle the training and test dataset
with open('train.pkl', 'wb') as file:
    pickle.dump(train, file)

with open('test.pkl', 'wb') as file:
    pickle.dump(test, file)

