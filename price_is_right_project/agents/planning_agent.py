from typing import Optional, List
from agents.agent import Agent
from agents.deals import Deal, DealSelection, Opportunity, ScrapedDeal
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from items import Item
from memory import Memory


class PlanningAgent(Agent):

    DEAL_THRESHOLD = 50

    name = "Planning Agent"
    color = Agent.GREEN


    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        super().__init__()
        self.initialize()
        Item.init_tokenizer()
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.ready()


    def run(self, deal: Deal) -> Opportunity:
        """
        Run the workflow for a particular deal
        :param deal: the deal, summarized from an RSS scrape
        :returns: an opportunity including the discount
        """
        self.log("Pricing potential deal")
        
        item = Item({'title': 'NA', 'description': [], 'features': [], 'details': deal.product_description}, 0)
        if item.prompt is None:
            item.make_prompt(item.details)

        estimate = self.ensemble.price(item)
        discount = estimate - deal.price
        self.log(f"Processed deal with discount ${discount:.2f}")

        return Opportunity(deal=deal, estimate=estimate, discount=discount)


    def plan(self, memory: Optional[Memory]) -> Optional[Opportunity]:
        """
        Run the full workflow:
        1. Use the ScannerAgent to find deals from RSS feeds
        2. Use the EnsembleAgent to estimate them
        3. Use the MessagingAgent to send a notification of deals
        :param memory: a list of Opportunities that have surfaced in the past
        :return: an Opportunity if one was surfaced, otherwise None
        """
        self.log("Starting run")

        selection = self.scanner.scan(memory=memory)
        if selection:
            opportunities = [self.run(deal) for deal in selection.deals[:5]]
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            best = opportunities[0]

            self.log(f"Identified best deal with discount ${best.discount:.2f}")

            if best.discount > self.DEAL_THRESHOLD:
                self.messenger.alert(best)

            self.log("Finished run")

            return best if best.discount > self.DEAL_THRESHOLD else None

        return None

