
import os
import time

import numpy as np

from algorithms.PaTSS import run_patss
from algorithms.ClaSP import run_clasp
from algorithms.FLOSS import run_floss

from data_handling.read_data import read_data
from data_handling.format_data import format_raw_values
#from visualization.visualize_segmentation import plot_segmentation
from evaluation.gradual import gradual_evaluation
from evaluation.floss import floss_evaluation
from evaluation.utility import convert_to_borders, convert_to_probabilistic_segmentation


def main(directory, algorithm, config, logger, dataset_name):
    """
    Execute a single algortihm on a single data set

    :param directory: The directory where the results should be saved
    :param algorithm: The algorithm to run
    :param config: The configuration of the algorithm with all hyper parameters and settings
    :param logger: The logger to write the results to
    :param dataset_name: The name of the data set to segment

    :return: The evaluation of the algorithm run and the time it took to run the algorithm
    """
    # Read the data
    start = time.time()
    (dataset_raw_values, ground_truth_raw_values) = read_data(config['data'])
    (multivariate_time_series, ground_truth) = format_raw_values(dataset_raw_values, ground_truth_raw_values)
    length_time_series = multivariate_time_series[0].shape[0]
    logger.write('>>> Read and formatted the data\n' +
                 'Length time series: %d\n' % length_time_series +
                 'Total time: %f seconds\n\n' % (time.time() - start))

    # Create subdirectories if requested
    if config['do_plotting'] and not os.path.exists(directory + '/figures/'):
        os.makedirs(directory + '/figures/')
    if config['write_segmentation'] and not os.path.exists(directory + '/segmentations/'):
        os.makedirs(directory + '/segmentations/')

    if algorithm == 'patss':
        # Run PaTSS if requested
        start_algorithm = time.time()
        segmentation, embedding, patterns = run_patss(directory, multivariate_time_series, length_time_series, config, logger)
        algorithm_execution_time = time.time() - start_algorithm
        # Compute the different formats of the segmentation
        algorithm_borders = convert_to_borders(segmentation, multivariate_time_series)
        probabilistic_segmentation = convert_to_probabilistic_segmentation(segmentation, multivariate_time_series)
        # Create and save a plot if requested
        if config['do_plotting']:
            start = time.time()
            plot_segmentation(directory + '/figures/', dataset_name, multivariate_time_series, embedding, patterns, segmentation, ground_truth['segmentation'])
            logger.write('>>> Successfully saved saved the segmentation plot\n' +
                         'Total time: %f seconds\n\n' % (time.time() - start))
        # Save the probability distribution if requested
        if config['write_segmentation']:
            start = time.time()
            np.savetxt(directory + '/segmentations/' + dataset_name + '.txt', segmentation)
            logger.write('>>> Successfully saved saved the segmentation\n' +
                         'Total time: %f seconds\n\n' % (time.time() - start))

    elif algorithm == 'clasp':
        # Run ClaSP if requested
        start_algorithm = time.time()
        algorithm_borders, clasp = run_clasp(multivariate_time_series, ground_truth, config, logger)
        algorithm_execution_time = time.time() - start_algorithm
        # Compute the probablistic segmentation (not used for final evaluation)
        probabilistic_segmentation = convert_to_probabilistic_segmentation(algorithm_borders, multivariate_time_series)
        # Create and save a plot if requested
        if config['do_plotting']:
            start = time.time()
            plot_segmentation(directory + '/figures/', dataset_name, multivariate_time_series, {}, {}, probabilistic_segmentation, ground_truth['segmentation'], clasp=clasp)
            logger.write('>>> Successfully saved saved the segmentation plot\n' +
                         'Total time: %f seconds\n\n' % (time.time() - start))
        # Save the borders computed by the algorithm if requested
        if config['write_segmentation']:
            start = time.time()
            np.savetxt(directory + '/segmentations/' + dataset_name + '.txt', algorithm_borders)
            logger.write('>>> Successfully saved saved the segmentation\n' +
                         'Total time: %f seconds\n\n' % (time.time() - start))

    # Run FLOSS if requested
    elif algorithm == 'floss':
        # Run FLOSS if requested
        start_algorithm = time.time()
        algorithm_borders, corrected_arc_crossings = run_floss(multivariate_time_series, ground_truth, config, logger)
        algorithm_execution_time = time.time() - start_algorithm
        # Compute the probablistic segmentation (not used for final evaluation)
        probabilistic_segmentation = convert_to_probabilistic_segmentation(algorithm_borders, multivariate_time_series)
        # Create and save a plot if requested
        if config['do_plotting']:
            start = time.time()
            plot_segmentation(directory + '/figures/', dataset_name, multivariate_time_series, {}, {}, probabilistic_segmentation, ground_truth['segmentation'], floss=(corrected_arc_crossings, algorithm_borders))
            logger.write('>>> Successfully saved saved the segmentation plot\n' +
                         'Total time: %f seconds\n\n' % (time.time() - start))
        # Save the borders computed by the algorithm if requested
        if config['write_segmentation']:
            start = time.time()
            np.savetxt(directory + '/segmentations/' + dataset_name + '.txt', algorithm_borders)
            logger.write('>>> Successfully saved saved the segmentation\n' +
                         'Total time: %f seconds\n\n' % (time.time() - start))

    else:
        raise Exception("Invalid algorithm given: '%s'!" % algorithm)

    start = time.time()
    evaluation = {}
    if 'floss' in config['evaluation_metric']:
        ground_truth_borders = convert_to_borders(ground_truth['segmentation'], multivariate_time_series)
        evaluation['floss'] = floss_evaluation(algorithm_borders, ground_truth_borders, length_time_series)
    if 'gradual' in config['evaluation_metric']:
        ground_truth_segmentation = convert_to_probabilistic_segmentation(ground_truth['segmentation'], multivariate_time_series)
        evaluation['gradual'] = gradual_evaluation(ground_truth_segmentation, probabilistic_segmentation, ground_truth['transition_areas'])
    logger.write(">>> Evaluated the algorithm\n" +
                 "Evaluation: %s\n" % evaluation +
                 'Total time: %f seconds\n\n' % (time.time() - start))

    return evaluation, algorithm_execution_time
