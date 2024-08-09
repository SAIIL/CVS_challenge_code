"""Define metrics for the CVS challenge. See compute_overall_metrics for specific types of metrics."""
from typing import Dict, List

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score)


def compute_overall_metrics(
    overall_labels: np.ndarray, confidence_aware_labels: np.ndarray, overall_confidences: np.array
) -> Dict:
    """Compute the metrics for accuracy and uncertainty calibration: f1 score, mean average precision, and accuracy for label accuracy, and Brier score for uncertainty calibration.

    Args:
        overall_labels (np.ndarray): An Nx3 labels array, where N is the number of samples, with 0 and 1 values. These are the ground truth labels.
        confidence_aware_labels (np.ndarray): An Nx3 labels array, where N is the number of samples, with 0 and 1 values. These are the ground truth labels.
        overall_confidences (np.ndarray): An Nx3 labels array, where N is the number of samples, with float values between 0 and 1. These are the model estimates of the label confidences.

    Returns:
        Dict: includes several metrics, such as F1, mAP, accuracy, and Brier score.
        For Brier Score: see Brier, Glenn W. 1950. “VERIFICATION OF FORECASTS EXPRESSED IN TERMS OF PROBABILITY.” Monthly Weather Review 78 (1): 1--3.
    """
    overall_predicted_labels = overall_confidences>0.5
    accuracy = accuracy_score(overall_labels, overall_predicted_labels)
    f1 = f1_score(
        overall_labels, overall_predicted_labels, average="samples", zero_division=1
    )  # 'samples' for multi-label setup
    mAP = average_precision_score(overall_labels, overall_confidences, average="macro")  # 'macro' for unweighted mAP
    metrics = {"f1": f1, "mAP": mAP, "accuracy": accuracy}
    metrics["brier_score"] = {}

    for i, key in enumerate(["c1", "c2", "c3"]):
        confidences = overall_confidences[:,i]
        ca_labels = confidence_aware_labels[:,i]
        # Use the original formula from the paper, as it works for continuous values
        brier_score = float(np.average((np.array(ca_labels)-np.array(confidences))**2))
        metrics["brier_score"][key] = brier_score

    return metrics

def compute_aggregate_dataset_metrics(dataset_result_dictionary: List):
    """Compute the metrics for the entire dataset.

    Args:
        dataset_result_dictionary (List): A list of dictionaries with the keys as the dataset names and the values as the results for that dataset.

    Returns:
        Dict: A dictionary with the keys as the dataset names and the values as the metrics for that dataset.
    """

    overall_raw_labels = None
    overall_outputs = None
    overall_confidence_aware_labels = None
    for dataset in dataset_result_dictionary:
        if overall_raw_labels is None:
            overall_raw_labels = dataset["overall_raw_labels"]
            overall_confidence_aware_labels = dataset["overall_confidence_aware_labels"]
            overall_outputs = dataset["overall_outputs"]
        else:
            overall_raw_labels = np.concatenate((overall_raw_labels, dataset["overall_raw_labels"]))
            overall_confidence_aware_labels = np.concatenate((overall_confidence_aware_labels, dataset["overall_confidence_aware_labels"]))
            overall_outputs = np.concatenate((overall_outputs, dataset["overall_outputs"]))

    overall_metrics = compute_overall_metrics(overall_raw_labels, overall_confidence_aware_labels, overall_outputs)
    overall_metrics['overall_raw_labels']=overall_raw_labels
    overall_metrics['overall_outputs']=overall_outputs
    overall_metrics['overall_confidence_aware_labels']=overall_confidence_aware_labels
    return overall_metrics
    
if __name__ == "__main__":
    # Test the metrics
    overall_labels = np.array([[0, 1, 0], [1, 0, 0],  [0, 0, 1]]).transpose()
    confidence_aware_labels = np.array([[0.2, 0.9, 0.2], [0.8, 0.0, 0.2],  [0.2, 0.0, 0.9]]).transpose()
    ## Perfectly calibrated confidences:
    # overall_confidences = np.array([[0.2, 0.9, 0.2], [0.8, 0.0, 0.2],  [0.2, 0.0, 0.9]]).transpose()
    ## Poorly calibrated confidences for c2, c3
    overall_confidences = np.array([[0.2, 0.9, 0.2], [0.50001, 0.4999, 0.4999],  [0.0, 0.0, 1.0]]).transpose()
    metrics = compute_overall_metrics(overall_labels=overall_labels, confidence_aware_labels=confidence_aware_labels,
                                overall_confidences=overall_confidences)
    print(metrics)

