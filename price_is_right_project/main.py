import modal

from agents.specialist_agent import SpecialistAgent

#Pricer = modal.Cls.from_name("pricer-service", "Pricer")
#pricer = Pricer()
#reply = pricer.price.remote(
#    "Keypad Door Knob with Key, Keyless Code Entry Lock, Auto Lock, 50 User Code, Easy to Install,"
#    " for Home ,Office, Hotel, Bedroom, Garage, No Deadbolt"
#)
#print(reply)

agent = SpecialistAgent()
agent.price(
    "Keypad Door Knob with Key, Keyless Code Entry Lock, Auto Lock, 50 User Code, Easy to Install,"
    " for Home ,Office, Hotel, Bedroom, Garage, No Deadbolt"
)

