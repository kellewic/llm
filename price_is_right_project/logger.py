import logging, os.path, sys
from datetime import datetime
from multiprocessing import RLock

## Add these here so users don't have to import logging just to pass these
## values to the init method.
CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

currentframe = lambda: sys._getframe(3)
_srcfile = os.path.normcase(currentframe.__code__.co_filename)

## Wrapper around Python logging module to perform routine items like creating
## log directory, formatting, and file/console logging depending on value of 
## variable.
class Logger(logging.Logger):
    ## Recursive lock to allow multiple processes to share the logger
    rlock = RLock()

    ## Thin wrappers that use the rlock
    def log(self, lvl, msg, *args, **kwargs):
        if self.closed is False:
            self.rlock.acquire()
            super().log(lvl, msg, *args, **kwargs)
            self.rlock.release()

    def critical(self, msg, *args, **kwargs):
        self.log(CRITICAL, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log(ERROR, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log(WARNING, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(INFO, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.log(DEBUG, msg, *args, **kwargs)


    ## Override this or the filename equals logger.py since
    ## that's where the actual log() method is called.
    def findCaller(self, stack_info=False, stacklevel=1):
        """
        Find stack frame of caller so we can note the source
        file name, line number, and function name.
        """
        f = currentframe()

        if f is not None:
            f = f.f_back

        rv = "(unknown file)", 0, "(unknown function)", None

        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)

            if filename == _srcfile:
                f = f.f_back
                continue

            rv = co.co_filename, f.f_lineno, co.co_name, None
            break

        return rv


    @classmethod
    def _get_name(cls, name: str|None=None) -> str:
        if name is None:
            ## Get the name of the script without the extension
            name = os.path.basename(sys.argv[0]).split('.')[0]

        return name


    def __init__(
        self,
        name:str|None = None,
        logdir:str|None = None,
        loglevel:int = logging.INFO,
        console:bool = False,
        stderr:bool = False,
        logThreads:bool = False,
        logProcesses:bool = False,
        logMultiprocessing:bool = False
    ):
        super().__init__(self._get_name(name))
        self.closed = False

        ## Assume (and force) console output only
        if logdir is None:
            console = True
        else:
            ## Ensure our logging directory exists
            try:
                os.mkdir(logdir, 0o775)
            except OSError as e:
                if e.args[1] != 'File exists':
                    print("Cannot create {logdir}: {e}")
                    sys.exit(1)

            logdir = f"{logdir.rstrip('/')}/"
            # File to where we send this script's logs
            logging_output_file = f"{logdir}{self.name}-{datetime.today().date()}.log"

        if console is True:
            if stderr is False:
                logging_output = sys.stdout
            else:
                logging_output = sys.stderr

            self.logfile = 'console'
        else:
            logging_output = open(logging_output_file, 'a')
            self.logfile = logging_output_file

        ## These enable/disable collecting of this log information
        logging.logThreads = logThreads
        logging.logProcesses = logProcesses
        logging.logMultiprocessing = logMultiprocessing

        self.setLevel(loglevel)

        logger_handler = logging.StreamHandler(logging_output)

        if logProcesses:
            logger_handler.setFormatter(logging.Formatter(
                "[%(asctime)s] %(name)s(%(process)d) %(levelname)s (%(filename)s "
                "=> %(lineno)s): %(message)s"
            ))
        else:
            logger_handler.setFormatter(logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s (%(filename)s "
                "=> %(lineno)s): %(message)s"
            ))

        self.logger_handler = logger_handler
        self.addHandler(logger_handler)


    def disable(self):
        self.removeHandler(self.logger_handler)

    def enable(self):
        self.addHandler(self.logger_handler)

    ## Close underlying handlers if they support the close() method call
    def close(self):
        self.closed = True

        for handler in self.handlers:
            if hasattr(handler, "close"): 
                handler.close()

