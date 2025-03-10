import numpy as np
from pathlib import Path
from data.datasets.ccd.utils import load_feature, get_video_paths  # Adjust for other datasets if needed


class ExplicitObjectDetector:
    """
    Handles explicit object detection using models like YOLO.
    - Loads the model once when instantiated.
    - Processes multiple videos without reloading.
    """

    def __init__(self, model_type="yolo"):
        self.model_type = model_type
        self.model = self.load_model()

    def load_model(self):
        """
        Placeholder for loading object detection model.
        (e.g., YOLOv8, Faster R-CNN)
        """
        print(f"Loading {self.model_type} model... (To be implemented)")
        return None  # Replace with actual model loading logic

    def detect_objects(self, video_path, max_objects=10):
        """
        Runs object detection model on the video.

        Args:
            video_path (str): Path to the video file.
            max_objects (int): Number of objects with highest confidence to keep.

        Returns:
            numpy.ndarray: Object detections in format (T, N, (x1, y1, x2, y2, prob, cls))
        """
        print(f"Running {self.model_type} object detection on {video_path}... (To be implemented)")
        detected_objects = None  # Replace with actual object detection logic
        return detected_objects


class ObjectDetectionExtractor:
    """
    Handles object detection extraction.
    - Uses precomputed detections if available.
    - Calls ExplicitObjectDetector only if needed.
    """

    def __init__(self, use_precomputed=True, model_type="yolo", max_objects=19):
        '''
        :param use_precomputed:
        :param model_type:
        :param max_objects: Only used if an explicit object detection model is used.
        '''
        self.use_precomputed = use_precomputed
        self.max_objects = max_objects
        self.detector = ExplicitObjectDetector(model_type) if not use_precomputed else None

    def extract_object_detections(self, video_path):
        """
        Extracts object detections for a given video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            numpy.ndarray: Object detections in format (T, N, (x1, y1, x2, y2, prob, cls))
        """

        if self.use_precomputed:
            # Load object detections from dataset utils
            detections = load_feature(video_path, feature_type="object_detection") # This is specific to CCD Dataset. Need to find a generalized approach
            if detections is not None:
                return detections
            else:
                print(f"No precomputed detections found for {video_path}.")
                return None

        # If explicit detection is enabled, run the object detection model
        if self.detector:
            return self.detector.detect_objects(video_path, max_objects=self.max_objects)

        return None  # Return None if nothing is available



if __name__ == "__main__":
    # Example usage

    paths = get_video_paths()

    video_path = paths[0]
    print(f"Extracting {video_path}...")

    # Using precomputed detections
    extractor = ObjectDetectionExtractor(use_precomputed=True)
    detections = extractor.extract_object_detections(video_path)
    print(f"Extracted object detections (precomputed): {detections.shape if detections is not None else None}")

    # Using explicit detection (To be implemented)
    # extractor = ObjectDetectionExtractor(use_precomputed=False, model_type="yolo")
    # detections = extractor.extract_object_detections(video_path)
    # print(f"Extracted object detections (explicit detection): {detections.shape if detections is not None else None}")
