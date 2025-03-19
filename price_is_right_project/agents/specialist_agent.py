import modal
from agents.agent import Agent

class SpecialistAgent(Agent):
    """
    An Agent that runs our fine-tuned LLM that's running remotely on Modal
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        """
        Set up this Agent by creating an instance of the modal class
        """
        super().__init__()
        self.initialize(" - connecting to Modal")
        
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()
        
        self.ready()
        
    def _price(self, description: str) -> float:
        """
        Make a remote call to return the estimate of the price of this item
        """
        self.log("Calling remote fine-tuned model")
        result = self.pricer.price.remote(description)
        self.log(f"Completed - predicting ${result:.2f}")

        return result
