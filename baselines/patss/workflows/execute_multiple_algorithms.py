
import os
import shutil
import tqdm
import workflows.execute_algorithm

from data_handling.read_data import get_all_data_set_names


def main(directory, config, logger):
    """
    Execute a workflow with multiple executions of an algorithm on multiple datasets.

    :param directory: The path of the directory to save the results in
    :param config: the config file for this experiment
    :param logger: The logger object to log all information
    """

    # Create a subdirectory for the figures (if necessary)
    if config['do_plotting']:
        os.makedirs(directory + '/figures/')

    # Create an overview file, i.e., a file containing the different algorithm runs and their results
    overview_file = open(directory + '/overview.csv', 'w')
    overview_file.write('data')
    if 'floss' in config['evaluation_metric']:
        overview_file.write(',floss')
    if 'gradual' in config['evaluation_metric']:
        overview_file.write(',gradual')
    overview_file.write(',time (s)\n')

    # Extract all datasets
    datasets_to_segment = []
    if 'datasets' in config.keys():
        for s in config['datasets']:
            datasets_to_segment.extend(get_all_data_set_names(s))
    if 'data' in config.keys():
        datasets_to_segment.extend(config['data'])

    # Setup the parameters
    exceptions = {}
    run_config = config['parameters'].copy()
    run_config['evaluation_metric'] = config['evaluation_metric']
    run_config['n_jobs'] = config['n_jobs'] if 'n_jobs' in config.keys() else 1
    run_config['write_segmentation'] = config['write_segmentation'] if 'write_segmentation' in config.keys() else False
    run_config['do_plotting'] = config['do_plotting']
    # Execute the algorithm on all datasets
    for dataset in tqdm.tqdm(datasets_to_segment):
        run_config['data'] = dataset
        dataset_name = dataset[:dataset.rfind('.')].replace('/', '_')

        logger.write('>>> Segmenting dataset %s\n' % dataset_name)
        run_logger = logger.create_sub_logger('individual_logs', '%s.txt' % dataset_name)
        try:
            os.makedirs(directory + '/temp/')

            # Evaluate the algorithm on the dataset. This is an 'execute_algorithm' workflow
            evaluations, algorithm_execution_time = workflows.execute_algorithm.main(directory, config['algorithm'], run_config, run_logger, dataset_name)
            # Save and log the evaluation
            overview_file.write(dataset)
            if 'floss' in config['evaluation_metric']:
                overview_file.write(',' + str(evaluations['floss']))
            if 'gradual' in config['evaluation_metric']:
                overview_file.write(',' + str(evaluations['gradual']))
            overview_file.write(',' + str(algorithm_execution_time) + '\n')
            logger.write('>>> Segmented dataset %s\n' % dataset_name)
            for item in evaluations.items():
                logger.write("%s: %s\n" % item)
            logger.write('Total time: %f seconds\n\n' % algorithm_execution_time)

        except Exception as e:
            run_logger.write("An exception occurred!")
            exceptions[dataset_name] = e
            if 'reraise_errors' in config.keys() and config['reraise_errors']:
                raise e

        finally:
            shutil.rmtree(directory + '/temp/')
            run_logger.close()

    overview_file.close()

    # Log all the failures
    if len(exceptions) > 0:
        logger.write('\n')
    for (data_set_with_exception, exception) in exceptions.items():
        logger.write(
            ">>> Exception in data set '%s': %s\n" % (data_set_with_exception, str(exception))
         )

    if len(exceptions) > 0:
        logger.write('\n')
        raise Exception(
            "In total %d exceptions occurred when executing multiple algorithms!\n" % len(exceptions) +
            "These occurred in the following datasets:\n - " +
            "\n - ".join(list(exceptions.keys()))
        )
