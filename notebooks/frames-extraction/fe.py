import cv2
import os
import argparse

def extract_frames(video_path, output_folder, fps=1):
    # make sure fps is a multiple of 0.2
    if (fps*10) % (0.2*10) != 0:  # *10 is to avoid floating-point precision issue
        raise ValueError("FPS must be a multiple of 0.2")

    # load video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError("Error opening video file")

    # get original fps
    original_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / fps)
    # print(original_fps, fps, frame_interval)

    # create output folder if not existed
    os.makedirs(output_folder, exist_ok=True)
    
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # extract subsets every frame_interval frame
        if frame_count % frame_interval == 0:
            output_frame_path = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(output_frame_path, frame)
            print(f"Extracted frame {extracted_count:04d}")
            extracted_count += 1
        
        frame_count += 1

    video.release()
    print("Done extracting frames.")


def process_videos_in_directory(videos_directory="videos/", frames_directory="frames/", fps=1):
    video_files = [f for f in os.listdir(videos_directory) if f.endswith(".mp4")]
    
    for video_file in video_files:
        video_path = os.path.join(videos_directory, video_file)
        video_name = os.path.splitext(video_file)[0]  # remove .mp4 extension
        output_folder = os.path.join(frames_directory, video_name)
        
        print(f"Processing video: {video_file}")
        extract_frames(video_path, output_folder, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos in batch mode")
    parser.add_argument("--videos_directory", type=str, default="videos/", help="Directory containing input video files")
    parser.add_argument("--frames_directory", type=str, default="frames/", help="Directory to save extracted frames")
    parser.add_argument("--fps", type=float, default=1, help="Frames per second at which to extract frames")

    args = parser.parse_args()
    process_videos_in_directory(args.videos_directory, args.frames_directory, args.fps)
