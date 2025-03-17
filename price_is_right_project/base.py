from logger import Logger

## Base class to wrap Logger
class Base():
    logger = None

    @classmethod
    def create_logger(cls):
        cls.logger = Logger(name=cls.__name__)

    def __init__(self):
        self.create_logger()

