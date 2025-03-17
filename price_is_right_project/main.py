#import modal

from agents import FrontierAgent, SpecialistAgent
from dotenv import find_dotenv, load_dotenv
from product_database import ProductDatabase

load_dotenv(dotenv_path=find_dotenv())

item_desc = (
    "Keypad Door Knob with Key, Keyless Code Entry Lock, Auto Lock, 50 User Code, Easy to Install,"
    " for Home ,Office, Hotel, Bedroom, Garage, No Deadbolt"
)

#db = ProductDatabase()
#collection = db.create_or_get_collection()

agent = FrontierAgent([])

#agent = SpecialistAgent()
#agent.price(item_desc)

