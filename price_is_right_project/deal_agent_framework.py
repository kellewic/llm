import numpy as np

from agents import PlanningAgent
from base import Base
from dotenv import find_dotenv, load_dotenv
from memory import Memory
from product_database import ProductDatabase
from sklearn.manifold import TSNE

# Colors for logging
BG_BLUE = '\033[44m'
WHITE = '\033[37m'
RESET = '\033[0m'

# Colors for plot
CATEGORIES = ['Appliances', 'Automotive', 'Cell_Phones_and_Accessories', 'Electronics','Musical_Instruments', 'Office_Products', 'Tools_and_Home_Improvement', 'Toys_and_Games']
COLORS = ['red', 'blue', 'brown', 'orange', 'yellow', 'green' , 'purple', 'cyan']

class DealAgentFramework(Base):

    def __init__(self):
        super().__init__()
        load_dotenv(dotenv_path=find_dotenv())

        self.collection = ProductDatabase().create_or_get_collection()
        self.memory = Memory()
        self.planner = None


    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing")
            self.planner = PlanningAgent(self.collection)
            self.log("Ready!")


    def log(self, message: str):
        self.logger.info(f'{BG_BLUE}{WHITE}{message}{RESET}')


    def run(self) -> Memory:
        self.log("Starting run")
        self.init_agents_as_needed()
        result = self.planner.plan(memory=self.memory)
        self.log(f"Finished run: {result}")

        return self.memory.write(result)


    @classmethod
    def get_plot_data(cls, max_datapoints=10000):
        collection = ProductDatabase().create_or_get_collection()
        result = collection.get(include=['embeddings', 'documents', 'metadatas'], limit=max_datapoints)
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        categories = [metadata['category'] for metadata in result['metadatas']]
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)

        return documents, reduced_vectors, colors


if __name__=="__main__":
    DealAgentFramework().run()

