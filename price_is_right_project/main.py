#import joblib, modal, pickle, re, sys

#import numpy as np

#from agents import EnsembleAgent, FrontierAgent, RandomForestAgent, SpecialistAgent
#from agents import DealSelection, ScrapedDeal, ScannerAgent
from agents import MessagingAgent, PlanningAgent, Deal
from dotenv import find_dotenv, load_dotenv
from items import Item
from product_database import ProductDatabase
#from testing import Tester

load_dotenv(dotenv_path=find_dotenv())

#with open('test.pkl', 'rb') as file:
#    test = pickle.load(file)

desc = """Best Buy has almost 350 TVs in the clearance and open box sale section. There is something for 
nearly any budget. We've pictured this Samsung DU6900 65" Crystal UHD 4K HDR Smart TV for $349.99 ($120 off.) 
Shipping is free for most items, but some are available for store pickup only. Best Buy has almost 350 TVs in 
the clearance and open box sale section. There is something for nearly any budget. We've pictured this Samsung 
DU6900 65" Crystal UHD 4K HDR Smart TV for $349.99 ($120 off.) Shipping is free for most items, but some are 
available for store pickup only. Buy Now at Best Buy
"""

#Item.init_tokenizer()
#item = Item({'title': 'NA', 'description': [], 'features': [], 'details': desc}, 0)
#print(item)


#db = ProductDatabase()
#collection = db.create_or_get_collection()

#planner = PlanningAgent(collection)
#planner.plan()

