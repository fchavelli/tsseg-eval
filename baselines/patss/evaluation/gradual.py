
import numpy as np


def gradual_evaluation(true_probabilities: np.ndarray, predicted_probabilities: np.ndarray, transition_areas: np.ndarray) -> float:
    """
    Compute the evaluation for gradual state transitions.

    :param true_probabilities: The ground truth probability distribution over the various semantic segments
    :param predicted_probabilities: The predicted probability distribution over the various semantic segments
    :param transition_areas: A boolean area indicating where the transitions occur

    :return: The MAE between the gradual ground truth and predicted probabilities in the transition areas
    """
    if transition_areas.shape[0] != predicted_probabilities.shape[1]:
        print("Error: transition_areas.shape[0] != predicted_probabilities.shape[1]")
        return np.nan

    # Iterate over the transition areas and compute the loss for each area
    nb_true_transitions = 0
    total_error = 0
    t = 0
    while t < transition_areas.shape[0]:
        if transition_areas[t]:
            start_transition = t
            while transition_areas[t]:
                t += 1
            end_transition = t - 1

            # Get the probabilities
            transition_probabilities_true = extract_state_probabilities(true_probabilities, start_transition, end_transition)
            transition_probabilities_predicted = extract_state_probabilities(predicted_probabilities, start_transition, end_transition)

            # Compute the loss for this transition area
            nb_true_transitions += 1
            total_error += np.mean(
                np.absolute(transition_probabilities_true[0, :] - transition_probabilities_predicted[0, :]) +
                np.absolute(transition_probabilities_true[1, :] - transition_probabilities_predicted[1, :])
            )
        else:
            t += 1

    return total_error / nb_true_transitions


def extract_state_probabilities(probabilities, start_transition, end_transition):
    """
    Extract the probabilities corresponding to the ending and starting semantic segment
    over the transition area. The ending semantic segment is identified as the segment with
    largest decrease in probability, while the starting segment is the one with highest increase
    in probability.

    :param probabilities: The probabilities over all the semantic segments
    :param start_transition: The start of the transition
    :param end_transition: The end of the transition

    :return: A 2D matrix containing the probabilities of the starting segment over the
             transition area in the first row and the ending segment in the second row
    """
    difference_probabilities = probabilities[:, start_transition] - probabilities[:, end_transition]
    segment_before = np.argmax(difference_probabilities)
    segment_after = np.argmin(difference_probabilities)
    return probabilities[(segment_before, segment_after), start_transition:end_transition]
