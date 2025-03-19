import json, math, os, re
import chromadb

from . import Agent
from datasets import load_dataset
from items import Item
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from testing import Tester
from typing import List, Dict

class FrontierAgent(Agent):

    MODEL = "gpt-4o-mini"

    name = "Frontier Agent"
    color = Agent.BLUE
    num_products = 5

    
    def __init__(self, collection):
        """
        Set up this instance by connecting to OpenAI or DeepSeek, to the Chroma Datastore,
        and setting up the vector encoding model.
        """
        super().__init__()
        self.initialize(" - connecting to model")

        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            self.client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
            self.MODEL = "deepseek-chat"
        else:
            self.client = OpenAI()
            self.MODEL = "gpt-4o-mini"

        self.log(f'Connected to {self.MODEL}')
        self.collection = collection
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.ready()


    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar products to the one being estimated
        :param prices: prices of the similar products
        :return: text to insert in the prompt that provides context
        """
        message = "For context, here are items similar to the item you must estimate.\n\n"

        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"

        return message


    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to OpenAI
        With the system and user prompt
        :param description: a description of the product
        :param similars: similar products to this one
        :param prices: prices of similar products
        :return: the list of messages in the format expected by OpenAI
        """
        system_message = "You estimate product prices. Reply only with the price, no explanation"

        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + description

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]


    def find_similars(self, description: str):
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """
        self.log(f"Search Chroma datastore for {self.num_products} similar products")

        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=self.num_products)
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]

        self.log(f"Found {self.num_products} similar products")
        return documents, prices


    def get_price(self, s) -> float:
        """
        A utility that plucks a floating point number out of a string
        """
        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0


    def _price(self, description: str) -> float:
        """
        Make a call to estimate the price of the described product by looking up
        similar products and including them in the prompt to give context.
        :param description: a description of the product
        :return: an estimate of the price
        """
        documents, prices = self.find_similars(description)

        self.log(f"Calling {self.MODEL} including context for {self.num_products} similar products")

        response = self.client.chat.completions.create(
            model=self.MODEL, 
            messages=self.messages_for(description, documents, prices),
            seed=42,
            max_tokens=5
        )

        reply = response.choices[0].message.content
        result = self.get_price(reply)

        self.log(f"Completed - predicting ${result:.2f}")
        return result

