import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from data.datasets.ccd.utils import load_feature, get_video_paths
from data.preprocessing.extract_frame_depth import MonocularDepth
from data.preprocessing.extract_object_detections import ObjectDetectionExtractor
from data.preprocessing.extract_object_depth import get_object_depth


class FeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', use_precomputed=True):
        """
        Initializes the feature extractor by loading required models.

        Args:
            device (str): 'cuda' or 'cpu' for running the models.
            use_precomputed (bool): Whether to use precomputed features if available.
        """
        self.device = device
        self.use_precomputed = use_precomputed

        # Load models
        self.depth_extractor = MonocularDepth(device=self.device)  # Frame depth extractor
        self.object_detector = ObjectDetectionExtractor(use_precomputed=self.use_precomputed)  # Object detection extractor

    def compute_object_depth(self, video_path):

        # Load Object Detections
        print(f"Loading object detections for {video_path}...")
        object_detections = self.object_detector.extract_object_detections(video_path)
        if object_detections is None:
            raise Exception(f"Object detections not found for {video_path}, unable to continue.")

        # Read Video Frames and Extract Frame Depth
        print(f"Processing video for depth extraction: {video_path}")

        # This is specific to CCD Dataset. Need to find a generalized approach
        path_from_current_file = Path(__file__).parent.parent / 'datasets' / 'ccd' / video_path

        video_capture = cv2.VideoCapture(str(path_from_current_file))
        frame_depths = []

        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(frame_count), desc=f"Extracting Depths for {video_path}"):
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert frame to PIL Image format and get depth
            frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_map = self.depth_extractor.find_frame_depth(frame_pil)
            frame_depths.append(depth_map)

        video_capture.release()
        frame_depths = np.array(frame_depths)  # Convert to numpy array


        # Compute Object Depths
        print(f"Computing object depths for {video_path}...")
        return get_object_depth(frame_depths, object_detections)

    def extract_features(self, video_path, feature_type="object_depth"):
        """
        Extracts features for a given video.

        Args:
            video_path (str): Path to the video file.
            feature_type (str): Type of feature to extract (currently supports 'object_depth').

        Returns:
            np.ndarray: Extracted object depth features.
        """

        # Check for Precomputed Features
        if self.use_precomputed:
            # This is specific to CCD Dataset. Need to find a generalized approach
            feature_data = load_feature(video_path, feature_type)
            if feature_data is not None:
                print(f"Using precomputed {feature_type} for {video_path}")
                return feature_data

        object_depths = self.compute_object_depth(video_path)

        print(f"Feature extraction complete for {video_path}. Returning object depths...")
        return np.array(object_depths)


if __name__ == "__main__":
    # Example Usage
    
    paths = get_video_paths()

    video_path = paths[0]
    print(f"Extracting {video_path}...")
    extractor = FeatureExtractor(use_precomputed=True)

    # Extract Object Depth Features
    obj_depth_features = extractor.extract_features(video_path, feature_type="object_depth")
    print("Extracted Object Depth Features Shape:", obj_depth_features.shape)
