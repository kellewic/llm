from logger import Logger

## Base class to wrap Logger
class Base():
    logger = None

    @classmethod
    def create_logger(cls):
        cls.logger = Logger(name=cls.__name__)

    @classmethod
    def disable_logging(cls):
        cls.logger.disable()

    @classmethod
    def enable_logging(cls):
        cls.logger.enable()

    def __init__(self):
        self.create_logger()

