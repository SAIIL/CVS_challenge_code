"""
The following is a simple example algorithm.
It is meant to run within a container.

To run it locally, you can call the following bash script:
  ./test_run.sh
This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:
  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.
Happy programming!
"""

from pathlib import Path
from resources.model import my_model

import cv2
import json
import numpy as np
import os


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
TMP_PATH = Path("/tmp") # NOTE temporary mounted storage for the container. This will not persist between foward passes, or into evaluation.
RESOURCE_PATH = Path("resources") # NOTE This is ideal for any additional code you need to store



def run():
    # Read the input
    # the video is located at '/input/laparoscopic-video.mp4'
    # here we are extracting all frames and convert them to a numpy array, 
    # feel free to read the video in differently though 
    # NOTE Only one 1 fps mp4 will be run at a time per container. 
    # Therfore only one mp4 will be available in the input_path location
    
    # The simplest way to adapt this example to your code would be to edit the block below
    # to integrate your model call

    ##########################
    # Begin Model Call
    ##########################
    # In this block, you should be able to control everything from your input pipeline (loading frames)
    # until the final output is generated.

    # Note carefully whether the input functions below load the frames as you expect them
    # e.g. RGB/BGR, 0-1/0-255 format.
    # Feel free to use a custom input function you feel comfartable with.

    input_frames = load_video_file_as_array(location=INPUT_PATH / "laparoscopic-video.mp4")

    _show_torch_cuda_info()

    # For now, let us set make bogus predictions: 1 video 3 frames with 3 criteria predictions each
    model_predictions = my_model(model_inputs=input_frames)

    output_cvs_criteria = {
        "overall_outputs": model_predictions
    }

    ##########################
    # End of Model Call
    ##########################
    # NOTE output_cvs_criteria should be a dictionary where the key "overall_outputs"
    # is of shape 90x3, corresponding to probability values for each of the 90 frames
    # between 0 and 1

    assert isinstance(output_cvs_criteria, dict), "Error: Your output_cvs_criteria should be a dictionary"
    assert "overall_outputs" in output_cvs_criteria.keys(), "Error: Your overall_outputs should be a key in the dictionary"
    assert np.shape(output_cvs_criteria["overall_outputs"]) == (90,3), "Error: You should be predicting 90 (frames) x 3 (criteria) probaility values in this run"
    assert check_elements_between_0_and_1(output_cvs_criteria["overall_outputs"]), "Error: Probability values should be between 0 and 1"

    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "cvs-criteria.json",
        content=output_cvs_criteria
    )

    return 0


def check_elements_between_0_and_1(list_of_lists):
    """
    Checks if all elements in a list of lists are within the range [0, 1].

    Args:
        list_of_lists (list of lists): A list where each element is a sublist, 
                                        and each sublist contains numerical elements.

    Returns:
        bool: Returns True if every element in every sublist is between 0 and 1 (inclusive),
              otherwise returns False.
    """
    return all(0 <= element <= 1 for sublist in list_of_lists for element in sublist)


def write_json_file(*, location, content):
    """
    Writes the provided content to a JSON file at the specified location.

    Args:
        location (str): The file path where the JSON file will be saved.
        content (dict or list): The content to be written to the JSON file. 
                                Should be a dictionary or list that is serializable to JSON.

    Returns:
        None: This function does not return any value.

    Functionality:
        - Serializes the `content` argument to JSON format with an indentation of 4 spaces.
        - Writes the JSON data to the file specified by the `location` argument.
    """
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def load_video_file_as_array(*, location):
    """
    Loads a video file and extracts frames at a rate of 1 frame per second (FPS). 
    The extracted frames are returned as a NumPy array.

    Args:
        location (str): The file path of the video to be loaded.

    Returns:
        np.ndarray: A NumPy array containing the extracted video frames. 
        Each frame is stored as an array of pixel values.
        None: If the video file cannot be opened or if an error occurs.

    Functionality:
        - Captures the video from the specified location.
        - Extracts frames at 1 FPS to ensure consistency with ground truth label mapping 
          during evaluation.
        - Stores the extracted frames in a list and converts it to a NumPy array.
        - Returns the array of frames, where each frame is a NumPy array of pixel values.

    Notes:
        - Videos are processed based on their original frame rate, but frames are extracted 
          at 1 FPS as required for evaluation.
        - Only 1 fps videos will be made available during the actual evaluation
        - The function ensures that only frames extracted at the specified interval are stored.
        - The returned NumPy array can be used for further processing or model input.

    """
    # Capture the video
    cap = cv2.VideoCapture(location)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
        # get original fps
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    # Note that the videos will all be at 1 fps by default

    frame_interval = int(original_fps / 1)
    frames = []
    frame_count = 0
    extracted_count = 0

    while True:
        # Read a frame
        ret, frame = cap.read()
        # If frame read was not successful, break the loop
        if not ret:
            break
        # Extract subsets every frame_interval frame and append the frame to the frames list
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            extracted_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()

    # Convert the list of frames to a NumPy array
    video_array = np.array(frames)
    return video_array



def extract_frames_as_png(*, location):
    """
    Extracts frames from a video file at a rate of 1 frame per second (FPS) and saves 
    them as PNG images in a specified folder. Note this function isn't used by default
    but you can use it if you prefer to read frames.

    Args:
        location (str): The file path of the video from which frames are to be extracted.

    Raises:
        IOError: If the video file cannot be opened.

    Functionality:
        - Loads a video from the specified location.
        - Extracts frames at 1 FPS to ensure consistent mapping with ground truth labels 
          (used for evaluation purposes).
        - Saves the extracted frames as PNG files in a folder named "extracted_frames".
        - The frames are saved as "frame_0000.png", "frame_0001.png", etc., based on the
          order in which they are extracted.

    Notes:
        - The video will be processed according to its original frame rate, but frames 
          are saved at 1 FPS as required for evaluation and submission.
        - Only 1 fps videos will be made available during the actual evaluation
        - The output folder "extracted_frames" will be created if it does not already exist.
        - The function prints a message each time a frame is successfully extracted.
    
    Example:
        extract_frames_as_png(location="path/to/video.mp4")

    """
    # load video
    video = cv2.VideoCapture(location)
    if not video.isOpened():
        raise IOError("Error opening video file")

    # get original fps
    original_fps = video.get(cv2.CAP_PROP_FPS)
    # Note that the videos will all be at 1 fps by default
    frame_interval = int(original_fps / 1)

    # create output folder if not existed
    extracted_frames_path = TMP_PATH / "extracted_frames"
    extracted_frames_path.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # extract subsets every frame_interval frame
        if frame_count % frame_interval == 0:
            file_path = str(extracted_frames_path / f"frame_{extracted_count:04d}.png")
            cv2.imwrite(file_path, frame)
            print(f"Extracted frame {extracted_count:04d}")
            extracted_count += 1
        frame_count += 1

    video.release()
    print("Done extracting frames.")

def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
