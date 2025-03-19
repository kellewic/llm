import joblib, modal
from agents.agent import Agent
from sentence_transformers import SentenceTransformer

class RandomForestAgent(Agent):
    name = "Random Forest Agent"
    color = Agent.MAGENTA

    def __init__(self):
        """
        Initialize this object by loading in the saved model weights 
        and the SentenceTransformer vector encoding model
        """
        super().__init__()
        self.initialize()
        self.vectorizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.model = joblib.load('random_forest_model.pkl')
        self.ready()
        
    def _price(self, description: str) -> float:
        """
        Use a Random Forest model to estimate the price of the described item
        :param description: the product to be estimated
        :return: the price as a float
        """
        self.log("Starting a prediction")
        vector = self.vectorizer.encode([description])
        result = max(0, self.model.predict(vector)[0])
        self.log(f"Completed - predicting ${result:.2f}")

        return result
