import json
from typing import Dict

import numpy as np


def save_results(
    estimated_detected_labels: np.ndarray, estimated_label_confidences: np.ndarray, output_filename: str
) -> None:
    """Save the resulting predictions and outputs over a whole dataset into a json

    Args:
        estimated_detected_labels (np.ndarray): N_samples x 3, values are 0 or 1, for whether c1..c3 have been detected.
        estimated_label_confidences (np.ndarray): N_samples x 3, values are continuous between 0 and 1, signifying confidence for c1..c3
        output_filename (str): Filename to save.
    """
    # Create and save the output structure
    output_file_content = {
        "estimated_detected_labels": estimated_detected_labels.tolist(),
        "estimated_label_confidences": estimated_label_confidences.tolist(),
    }

    # Save results file
    # This is an example file, similar to the one you would need to generate to save results to be evaluated.
    with open(output_filename, "w") as fp:
        json.dump(output_file_content, fp)


def load_results(filename: str) -> Dict:
    """Load the results from the json filename.

    Args:
        filename (str): A result json with labels and confidences.

    Returns:
        Dict: A dictionary with two np.ndarray values - estimated labels and estimated confidences, each of size N_samples x 3, where N_samples is the dataset size.
    """
    with open(filename, "r") as fp:
        file_content = json.load(fp)
    for key in ["estimated_detected_labels", "estimated_label_confidences"]:
        file_content[key] = np.array(file_content[key])
    return file_content
