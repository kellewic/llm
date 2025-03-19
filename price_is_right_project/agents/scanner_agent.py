import json, os

from . import Agent
from .deals import DealSelection, ScrapedDeal
from openai import OpenAI
from typing import List, Optional

class ScannerAgent(Agent):

    name = "Frontier Agent"
    color = Agent.CYAN

    MODEL = "gpt-4o-mini"

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list.
	You select deals with high quality descriptions and the clearest price that is greater than zero.
	Provide the price as a number derived from the description.
	If the price of a deal isn't clear, do not include that deal in your response.
    Most important is that you respond with exactly 5 deals with the most detailed product description and price.
	It's not important to mention deal terms; a thorough product description is most important.
    Be careful with products described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product.
	Only respond with products where you are highly confident of the price. 
	Respond strictly in JSON with no explanation, using the following format:

    {"deals": [
        {
            "product_description": "4-5 sentence product summary focusing on item details rather than why it's a good deal. Do not mention discounts or coupons.",
            "price": 99.99,
            "url": "the url as provided"
        },
        ...
    ]}"""

    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list.
    
    Deals:
    
    """

    
    def __init__(self):
        """
		Set up this instance by initializing OpenAI
        """
        super().__init__()
        self.initialize()
        self.openai = OpenAI()
        self.ready()


    def fetch_deals(self, memory:List[str]=[]) -> List[ScrapedDeal]:
        """
        Look up deals published on RSS feeds
        Return any new deals that are not already in the memory provided
        """
        self.log("Fetch deals from RSS feed")
        urls = [opp.deal.url for opp in memory]
        scraped = ScrapedDeal.fetch(show_progress=True)
        result = [scrape for scrape in scraped if scrape.url not in urls]
        self.log(f"Received {len(result)} deals not already scraped")

        return result


    def make_user_prompt(self, scraped) -> str:
        """
        Create a user prompt for OpenAI based on the scraped deals provided
        """
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += '\n\n'.join([scrape.describe() for scrape in scraped])

        return user_prompt


    def scan(self, memory: List[str]=[]) -> Optional[DealSelection]:
        """
        Call OpenAI to provide a high potential list of deals with good descriptions and prices
        Use StructuredOutputs to ensure it conforms to our specifications
        :param memory: a list of URLs representing deals already raised
        :return: a selection of good deals, or None if there aren't any
        """
        scraped = self.fetch_deals(memory)

        if scraped:
            self.log("Calling OpenAI using structured output")

            result = self.openai.beta.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.make_user_prompt(scraped)}
                ],
                response_format=DealSelection
            )
            result = result.choices[0].message.parsed
            result.deals = [deal for deal in result.deals if deal.price>0]

            self.log(f"Received {len(result.deals)} selected deals with price>0")

            return result

        return None

