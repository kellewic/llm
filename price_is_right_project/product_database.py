import chromadb, pickle

from base import Base
from logger import Logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class ProductDatabase(Base):
    collection_name = "products"

    def __init__(self):
        super().__init__()
        self.logger.info("Creating chromadb.PersistentClient")
        settings = chromadb.config.Settings(allow_reset=True)
        self.client = chromadb.PersistentClient(path="products_vectorstore", settings=settings)


    def reset(self):
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Collection <{self.collection_name}> deleted")
        except Exception as e:
            pass


    def description(self, item):
        text = item.prompt.replace("How much does this cost to the nearest dollar?\n\n", "")
        return text.split("\n\nPrice is $")[0]


    def create_or_get_collection(self):
        collection = None

        try:
            collection = self.client.get_collection(self.collection_name)
            self.logger.info("Retrieving collection")
        except:
            self.logger.info("Creating collection")
            collection = self.client.create_collection(self.collection_name)

            train = None
            with open('train.pkl', 'rb') as file:
                train = pickle.load(file)

            train = train[:1000]

            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            step = 1000

            for i in tqdm(range(0, len(train), step)):
                docs_len = i + step

                documents = [self.description(item) for item in train[i: docs_len]]
                vectors = model.encode(documents).astype(float).tolist()
                metadatas = [{"category": item.category, "price": item.price} for item in train[i: docs_len]]
                ids = [f"doc_{j}" for j in range(i, docs_len)]

                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=vectors,
                    metadatas=metadatas
                )

        self.logger.info(f"Collection has {collection.count():,} items")
        return collection
