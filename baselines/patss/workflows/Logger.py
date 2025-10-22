
import os


"""
A class that handles the logging of the process of different algorithm executions. 
"""
class Logger:

    def __init__(self, do_logging: bool, verbose: bool, log_file_directory: str, log_file_name: str):
        """
        Initialize the logger.

        :param do_logging: Whether the process should be logged to a log-file.
        :param verbose: Whether the process should be printed to the terminal.
        :param log_file_directory: The directory for the log files.
        :param log_file_name: The name of the log file.
        """
        if do_logging:
            self.__log_file = open(log_file_directory + '/' + log_file_name, 'w')
        else:
            self.__log_file = None
        self.__do_logging = do_logging
        self.__log_file_directory = log_file_directory
        self.__verbose = verbose

    def write(self, message: str):
        """
        Write a message to the log file (if do_logging is True) and to the terminal
        (if verbose is True)

        :param message: The message to write.
        """
        if self.__do_logging:
            self.__log_file.write(message)

        if self.__verbose:
            print(message)

    def create_sub_logger(self, sub_directory: str, log_file_name: str):
        """
        Create a sub logger from this logger with the same do_logging and verbose
        settings. The logger will write its log files to the sub directory of the
        log file directory of this logger.

        :param sub_directory: The name of the subdirectory to write the log files to.
        :param log_file_name: The name for the log file of the sub logger.

        :return: A sub logger derived from this Loggger
        """
        full_sub_directory = self.__log_file_directory + '/' + sub_directory
        if self.__do_logging and not os.path.isdir(full_sub_directory):
            os.mkdir(full_sub_directory)
        return Logger(self.__do_logging, self.__verbose, full_sub_directory, log_file_name)

    def close(self):
        """
        Close this logger, i.e., the log file if it is being used.
        """
        if self.__do_logging:
            self.__log_file.close()
