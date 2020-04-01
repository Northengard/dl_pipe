import os
import sys
import logging
from io import StringIO
from .storage import _LOG_DIR


class Logger:
    def __init__(self, exp_name, filename, stderr=False):
        """
        Logger to forward all output to file
        :param exp_name:
        :param filename:
        :param stderr:
        """
        self.terminal = sys.stdout if not stderr else sys.stderr
        dir_name = os.path.join(_LOG_DIR, exp_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        self.log = open(os.path.join(dir_name, filename), "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()

        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


class TqdmToLogger(StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
        """
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


def get_logger(logger_path):
    """
    create logfile and return logger by given log path using logging lib
    :param logger_path: str, path to log file
    :return: logger
    """
    # create logger for prd_ci
    logger_name = os.path.basename(logger_path)
    log = logging.getLogger(logger_name)
    log.setLevel(level=logging.DEBUG)

    # create file handler for logger.
    fh = logging.FileHandler(logger_path)
    fh.setLevel(level=logging.DEBUG)

    log.addHandler(fh)
    return log
