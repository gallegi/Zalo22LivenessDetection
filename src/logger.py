import logging
import os
import warnings
import traceback

class Logger():
    def __init__(self, logger_name, file_path, mode='a'):
        if os.path.exists(file_path):
            warnings.warn(f'Log file already exists at {file_path}. It will be overwritten.')
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        self.logger = logging.getLogger(logger_name)
        if not self.logger.hasHandlers():
            # set log level
            self.logger.setLevel(logging.INFO)

            # define file handler and set formatter
            file_handler = logging.FileHandler(file_path, mode=mode)
            formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
            file_handler.setFormatter(formatter)

            # add file handler to logger
            self.logger.addHandler(file_handler)
    
    def log_and_print_object(self, obj):
        print(obj)
        self.logger.info(obj)

    def log_error(self, cmd_out=True):
        error_trace = traceback.format_exc()
        if cmd_out:
            print(error_trace)
        self.logger.error(error_trace)

    def log_warning(self, warn_msg):
        warnings.warn(warn_msg)
        self.logger.warning(warn_msg)
        
    def log_tqdm(self, tqdm_iter):
        str_bar_msg = str(tqdm_iter)
        # self.logger.info(str_bar_msg) 

    
if __name__ == '__main__':
    print(1)
    log_name = 'log1'
    path = f'./{log_name}.txt'
    logger = Logger(log_name, path)
    logger.log_and_print_object('abc')
    # it = tqdm.tqdm(range(100), bar_format='{l_bar}{bar:70}{r_bar}{bar:-70b}')
    # for i in it:
    #     if i % 10 == 0:
    #         # logger.log_and_print_object(i)
    #         pass
    #     time.sleep(0.01)
    # logger.log_tqdm(it)
    # it.close()

    try:
        1 / 0
    except Exception as ex:
        logger.log_error()

    logger.log_and_print_object('continue prog')

    logger.log_warning('Test warning log')