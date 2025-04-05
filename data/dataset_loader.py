# Currently this file is specifically written for ccd dataset. Should be updated to generalize for all datasets

import torch
import numpy as np
from torch.utils.data import Dataset
from data.datasets.ccd.utils import load_annotation, get_video_paths, load_feature, get_n_frames


class VideoDataset(Dataset):
    """
    Custom PyTorch Dataset for Accident Anticipation.

    Loads precomputed features including:
    - Object detections
    - Object depth
    - Frame-level and object-level embeddings (VGG)
    - Binary accident labels (per frame)
    - Time-To-Accident (TTA) labels

    Args:
        is_train (bool): If True, uses training data; otherwise, uses test data.
    """

    def __init__(self, is_train=True):
        self.annotations = load_annotation()
        self.is_train = is_train
        self.split = "train" if is_train else "test"
        self.paths = get_video_paths(split=self.split)
        self.n_frames = get_n_frames()

    def __len__(self):
        return len(self.paths)

    def _get_tta(self, bin_labels):
        """
        Computes Time-To-Accident (TTA) for positive samples using NumPy.

        Args:
            bin_labels (list or np.array): Binary accident labels (0 = no accident, 1 = accident occurring).

        Returns:
            list: TTA values for each frame.
        """
        bin_labels = np.array(bin_labels)  # Convert to NumPy array (if not already)

        # Find the first occurrence of '1' (accident frame)
        accident_idx = np.where(bin_labels == 1)[0][0]

        # Compute TTA for all frames
        return np.maximum(0, accident_idx - np.arange(len(bin_labels)))


    def __getitem__(self, index):
        """
        Loads features and labels for a given video.

        Args:
            index (int): Index of the video sample.

        Returns:
            tuple of torch.Tensor: (
                object_detections,  # (T, N, 6)
                object_depths,  # (T, N, 1)
                object_features,  # (T, N, D)
                frame_features,  # (T, 1, D)
                bin_labels,  # (T,)
                tta,  # (T,)
                cls  # (1,)
            )
        """
        path = self.paths[index]
        video_name = path.stem
        cls = 1 if "positive" in str(path) else 0  # 1 = Accident, 0 = No Accident

        # Load features
        object_detections = load_feature(video_path=path, feature_type="object_detection")
        object_depths = load_feature(video_path=path, feature_type="object_depth")
        features = load_feature(video_path=path, feature_type="VGG")

        # Split features into frame-level and object-level
        frame_features, object_features = features[:, 0, :], features[:, 1:, :]

        # Load binary accident labels
        bin_labels = self.annotations[video_name] if cls==1 else [0] * self.n_frames

        # Compute TTA
        if cls == 1:
            tta = self._get_tta(bin_labels)
        else:
            tta = [self.n_frames + 1] * self.n_frames  # Large TTA for negative samples

        # Convert to tensors
        if self.is_train:
            return (
                torch.tensor(object_detections, dtype=torch.float32),
                torch.tensor(object_depths, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(object_features, dtype=torch.float32),
                torch.tensor(frame_features, dtype=torch.float32).unsqueeze(1),
                torch.tensor(bin_labels, dtype=torch.float32),
                torch.tensor(tta, dtype=torch.float32),
                torch.tensor(cls, dtype=torch.float32).unsqueeze(-1),
            )
        else:
            return (
                torch.tensor(object_detections, dtype=torch.float32),
                torch.tensor(object_depths, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(object_features, dtype=torch.float32),
                torch.tensor(frame_features, dtype=torch.float32).unsqueeze(1),
                torch.tensor(bin_labels, dtype=torch.float32),
                torch.tensor(tta, dtype=torch.float32),
                torch.tensor(cls, dtype=torch.float32).unsqueeze(-1),
                video_name
            )


if __name__ == '__main__':
    dataset = VideoDataset(is_train=False)
    object_detections, object_depths, object_features, frame_features, bin_labels, tta, cls = dataset[0]
    print(object_detections.shape)
    print(object_depths.shape)
    print(object_features.shape)
    print(frame_features.shape)
    print(bin_labels.shape)
    print(tta.shape)
    print(cls.shape)
