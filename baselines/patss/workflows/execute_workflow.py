
import os
import argparse
import json
import time
import shutil
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

import workflows.execute_algorithm
import workflows.execute_multiple_algorithms
from workflows.Logger import Logger

from data_handling.read_data import set_data_directory
from visualization.visualize_segmentation import update_runtime_configuration_parameters


def main(data_directory, experiment_directory, experiment):
    """
    The main loop of this project, for executing the different workflows. This works through
    experiments, i.e., a directory with a config.json file containing the configuration.

    :param data_directory: The directory containing all the data
    :param experiment_directory: The directory containing the experiments
    :param experiment: The name of the experiment directory to execute, within the
                       experiment_directory
    """
    # Set the data directory
    set_data_directory(data_directory)

    # Check if valid experiment is given, that is the experiment directory exists and contains a config.json file
    # If the experiment is not valid, all subdirectories will be executed recursively as experiment
    experiment_path = experiment_directory + '/' + experiment
    if os.path.isfile(experiment_path + '/config.json'):
        print('Execute experiment: %s' % experiment)
    elif os.path.isdir(experiment_path):
        for file in os.listdir(experiment_path):
            if os.path.isdir(experiment_path + '/' + file):
                main(data_directory, experiment_path, file)
        return 0
    else:
        print('Invalid experiment given: %s!' % experiment)
        return -1

    # Clean up the experiment directory
    for file in os.listdir(experiment_path):
        if os.path.isdir(experiment_path + '/' + file):
            shutil.rmtree(experiment_path + '/' + file)
        elif file != 'config.json':
            os.remove(experiment_path + '/' + file)

    # Initialize the configuration and logger
    with open(experiment_path + '/config.json', 'r') as config_file:
        config = json.load(config_file)
    logger = Logger(config['do_logging'], config['verbose'], experiment_path, 'log.txt')

    # Set plotting settings:
    if 'plotting_configuration' in config.keys():
        update_runtime_configuration_parameters(config['plotting_configuration'])

    try:
        # Start the time
        start_time = time.time()

        # Execute the requested workflow
        if config['workflow-type'] == 'algorithm_execution':
            if config['do_plotting']:
                os.makedirs(experiment_path + '/figures/')
            run_config = config['parameters']
            run_config['n_jobs'] = config['n_jobs'] if 'n_jobs' in config.keys() else 1
            run_config['write_segmentation'] = config['write_segmentation'] if 'write_segmentation' in config.keys() else False
            run_config['do_plotting'] = config['do_plotting']
            run_config['data'] = config['data']
            run_config['evaluation_metric'] = config['evaluation_metric'] if 'evaluation_metric' in config.keys() else []
            os.makedirs(experiment_path + '/temp/')
            workflows.execute_algorithm.main(experiment_path, config['algorithm'], run_config, logger, config['data'][:config['data'].rfind('.')].replace('/', '_'))
            shutil.rmtree(experiment_path + '/temp/')

        elif config['workflow-type'] == 'multiple_algorithms_execution':
            workflows.execute_multiple_algorithms.main(experiment_path, config, logger)

        else:
            raise Exception("Invalid workflow type given '%s'!" % config['workflow-type'])

        # Log successful execution of the workflow
        logger.write(
            '==================================================================\n'
            'Workflow successfully executed!\n'
            'Total time required: %f seconds\n' % (time.time() - start_time))

    except Exception as e:
        # Log failure execution of the workflow
        logger.write("An exception occurred during execution: '%s'\n" % str(e))
        raise e

    finally:
        logger.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Pattern Based Time Series Clustering.")
    parser.add_argument('--experiment', dest='experiment',
                        type=str, default='test_univariate_algorithm_execution',
                        help='The name of the experiment to execute')
    parser.add_argument('--experiment_dir', dest='experiment_dir',
                        type=str, default='../experiments/',
                        help='The directory containing the experiment files')
    parser.add_argument('--data_dir', dest='data_dir',
                        type=str, default='../data/',
                        help='The directory containing the datasets')

    args = parser.parse_args()
    main(args.data_dir, args.experiment_dir, args.experiment)
