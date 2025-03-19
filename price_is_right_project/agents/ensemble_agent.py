import joblib

import pandas as pd

from .agent import Agent
from .frontier_agent import FrontierAgent
from .random_forest_agent import RandomForestAgent
from .specialist_agent import SpecialistAgent

from items import Item

class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        """
        Create an instance of Ensemble, by creating each of the models 
        and loading the weights of the Ensemble
        """
        super().__init__()
        self.initialize()
        
        self.frontier_agent = FrontierAgent(collection)
        self.random_forest_agent = RandomForestAgent()
        self.specialist_agent = SpecialistAgent()

        self.agents = [self.specialist_agent,self.frontier_agent, self.random_forest_agent]
        self.model = joblib.load('ensemble_model.pkl')

        self.ready()
        
    def price(self, item: Item) -> float:
        """
        """
        self.log("Starting a prediction")
        result = 0

        df = {}
        agent_prices = []
        for agent in self.agents:
            name = type(agent).__name__
            self.log(f"Collaborating with {name}")
            price = agent.price(item)

            df[name.replace('Agent', '')] = [price]
            agent_prices.append(price)

        df['Min'] = [min(agent_prices)]
        df['Max'] = [max(agent_prices)]

        X = pd.DataFrame(df)
        result = max(0, self.model.predict(X)[0])

        self.log(f"Completed - predicting ${result:.2f}")

        return result
