import os
import logging
import datetime


class Logger:
    def __init__(self, directory=None, name="model.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # formatter = logging.Formatter('[%(levelname)s]\t%(asctime)s\n%(message)s')
        formatter = logging.Formatter('%(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        if directory is not None:
            file_handler = logging.FileHandler(os.path.join(directory, name))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def __call__(self, *args, **kwargs):
        message = [str(arg) for arg in args]
        message = "\n".join(message)
        self.logger.info(message)

    def time(self):
        message = "Time\t{}".format(datetime.datetime.now())
        self.logger.info(message)

    def line(self, character="-", length=80):
        message = "".join([character for i in range(length)])
        self.logger.info(message)


if __name__ == "__main__":
    logger = Logger()
    logger("HI", "Hello")
    logger.time()
    logger.line()