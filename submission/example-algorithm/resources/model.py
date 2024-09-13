# NOTE the import structure for various resources used by my model.
from .constants import CONTENT_PATH
from .utils.file_ops import read_text_file

import numpy as np


def my_model(*, model_inputs):
    """
    Example model that generates predictions for a batch of input images.

    The function takes image inputs of shape (n, h, w, c) and returns
    a dictionary containing the predicted class probabilities for each image. 
    Each image receives a prediction consisting of three probability values 
    representing the likelihood of belonging to one of three classes.

    Parameters:
    -----------
    model_inputs : array-like
        A NumPy array or tensor of shape (n, h, w, c) representing a batch of images, where:
        - n: Number of images (batch size).
        - h: Height of each image.
        - w: Width of each image.
        - c: Number of channels in each image (e.g., 3 for RGB images).
    
    Returns:
    --------
    predictions : dict
        A dictionary with the following key:
        - "overall_outputs": A list of lists with shape (n, 3), where each sub-list 
          contains three probability values (for three classes) corresponding to 
          each input image. Each set of probabilities sums to 1, representing the 
          likelihood of the image belonging to each of the three classes.
    
    Example:
    --------
    >>> predictions = my_model(model_inputs=model_inputs)
    >>> print(predictions["overall_outputs"])
    [[0.2, 0.56, 0.75], [0.57, 0.75, 0.75], [0.78, 0.9, 0.9]]
    
    """
    print(read_text_file(CONTENT_PATH))
    print('Making predictions on model inputs')

    ##############
    # Your Model 
    ##############
    n,_,_,_ = np.shape(model_inputs)

    predictions = np.random.rand(n,3).tolist()

    return predictions
