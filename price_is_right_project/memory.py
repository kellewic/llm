import json, os
from typing import List, Optional

from agents import Opportunity

class Memory:
    MEMORY_FILENAME = "memory.json"

    def __init__(self):
        self.memory = self.read()


    def __iter__(self):
        for opp in self.memory:
            yield opp


    def read(self) -> List[Opportunity]:
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as file:
                data = json.load(file)

            return [Opportunity(**item) for item in data]

        return []


    def write(self, opp: Optional[Opportunity], flush: bool=True) -> List[Opportunity]:
        if opp is not None:
            self.memory.append(opp)

            if flush is True:
                data = [opportunity.dict() for opportunity in self.memory]
                with open(self.MEMORY_FILENAME, "w") as file:
                    json.dump(data, file, indent=2)

        return self.memory

