import logging


class myLogs:
    '''

    '''
    def __init__(self, level=logging.INFO, filename="mylogs.log"):
        logLevel = level
        consoleFormat = '%(funcName)s[line:%(lineno)d] %(levelname)s %(message)s'
        fileFormat = '%(asctime)s %(filename)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s'
        logging.basicConfig(level=logLevel,
                            format=fileFormat,
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=filename,
                            filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logLevel)
        formatter = logging.Formatter(consoleFormat)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        self.logger = logging.getLogger('')
        self.logger.propagate = True


