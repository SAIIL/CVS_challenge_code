import glob
import os
from statistics import mode

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CVSData(Dataset):
    """Dataloader that returns annotated frames and their
    corresponding labels (majority), video names, and frame ids.
      Args:
        frames_path (str):   Path to folder containing the extracted frames
        labels_path (str):   Path to folder containing a label csv file for each video

      Returns:
        img, label, video_name, frame_id, metadata
    """

    def __init__(self, frames_path, labels_path, transform=None):
        self.frames_path = frames_path
        self.labels_path = labels_path
        self.transform = transform
        search_pattern = os.path.join(frames_path, "**", "*.jpg")
        frames = sorted(glob.glob(search_pattern, recursive=True))
        self.image_paths = [file for file in frames if self.is_annotated(file)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        video_name = img_path.split("/")[-2]
        frame_id = int(img_path.split("/")[-1].replace(".jpg", "")) - 1
        label_path = os.path.join(self.labels_path, f"{video_name}/frame.csv")
        video_label_path = os.path.join(self.labels_path, f"{video_name}/video.csv")

        # Load image
        img = Image.open(img_path)

        # Load c1,c2,c3 label
        label_df = pd.read_csv(label_path)
        video_label_df = pd.read_csv(video_label_path)
        confidences = [float(video_label_df[f'confidence_rater{i+1}'].iloc[0]) for i in range(3)]
        label_df = label_df[label_df["frame_id"] == frame_id]
        c1, c2, c3 = (
            self.majority_vote(label_df, "c1"),
            self.majority_vote(label_df, "c2"),
            self.majority_vote(label_df, "c3"),
        )
        ca_c1 = self.confidence_multiplexed_majority_vote(label_df, "c1",confidences)
        ca_c2 = self.confidence_multiplexed_majority_vote(label_df, "c2",confidences)
        ca_c3 = self.confidence_multiplexed_majority_vote(label_df, "c3",confidences)
        label = torch.as_tensor([c1, c2, c3], dtype=torch.float32)

        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)
        metadata = {}
        metadata["raw_labels"] = {}
        for key in ["c1", "c2", "c3"]:
            metadata["raw_labels"][key] = self.raw_labels(label_df, key)
        metadata["confidence_aware_labels"] = {}
        ca_labels = [ca_c1,ca_c2,ca_c3]
        for i,key in enumerate(["c1", "c2", "c3"]):
            metadata["confidence_aware_labels"][key] = ca_labels[i]
        
        return img, label, video_name, frame_id, metadata

    def is_annotated(self, fname):
        frame_id = int(os.path.splitext(os.path.basename(fname))[0]) - 1
        return frame_id % 150 == 0

    def raw_labels(self, row, category):
        labels = [
            row[f"{category}_rater1"].iloc[0],
            row[f"{category}_rater2"].iloc[0],
            row[f"{category}_rater3"].iloc[0],
        ]

        return labels

    def majority_vote(self, row, category):
        labels = [
            row[f"{category}_rater1"].iloc[0],
            row[f"{category}_rater2"].iloc[0],
            row[f"{category}_rater3"].iloc[0],
        ]
        return mode(labels)

    def confidence_multiplexed_majority_vote(self, row, category,confidences):
        labels = np.array([
            row[f"{category}_rater1"].iloc[0],
            row[f"{category}_rater2"].iloc[0],
            row[f"{category}_rater3"].iloc[0],
        ])
        labeler_confidences = np.array(confidences)
        confidence_aware_label = 0.5+1/3.0*np.dot(labels-0.5,labeler_confidences)
        return confidence_aware_label
