# Video Frame Extractor

This script extracts frames from videos in the `videos/` directory or a customized input directory at a specified frames-per-second (FPS) rate and saves them to corresponding folders in the `frames/` directory or a customized output directory.

## Requirements

- Python 3.x
- OpenCV

## Installation

1. Clone the repository or download the script.
2. Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

To run the script, use the following command:

```bash
python fe.py --videos_directory <path_to_videos_directory> --frames_directory <path_to_frames_directory> --fps <fps>
```

Arguments

- `--videos_directory`: Directory containing input video files. Defaults to `videos/`.
- `--frames_directory`: Directory to save extracted frames. Defaults to `frames/`.
- `--fps`: Frames per second at which to extract frames. Defaults to 1. The FPS value must be a multiple of 0.2..

Example

```bash
python fe.py --videos_directory videos --frames_directory frames --fps 1.2
```

This command will extract frames from all videos in the `videos` directory at 1.2 frame per second and save them in corresponding folders within the `frames` directory.

If you want to use all default parameters, you can simply run:

```bash
python fe.py
```

This command will use `videos/` as the input directory, `frames/` as the output directory, and extract frames at 1 frame per second.

## Notes

- Ensure that the specified FPS value is a multiple of 0.2 to avoid errors.
- The script will create the output folders if they do not exist.
- Each video will have its frames saved in a folder named after the video file (without the `.mp4` extension) in the `frames` directory.
- Frames are saved in the JPEG format with filenames frame_0000.jpg, frame_0001.jpg, etc.