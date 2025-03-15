# Runs the full preprocessing pipeline for the dataset
import time
import logging
from tqdm import tqdm
from pathlib import Path

from data.datasets.ccd.utils import get_video_paths, save_feature
from data.preprocessing.extract_features import FeatureExtractor


class PreprocessDataset:
    def __init__(self, mode="test", start_video_number=0, end_video_number=None, use_precomputed=True):
        """
        Initializes the dataset preprocessing class.

        Args:
            mode (str): Dataset split to process ('train' or 'test').
            start_video_number (int): Starting index for video processing.
            end_video_number (int): Ending index for video processing.
            use_precomputed (bool): Whether to use precomputed features if available.
        """
        self.mode = mode

        self.video_paths = get_video_paths(split=self.mode)

        self.start_video_number = start_video_number
        self.end_video_number = end_video_number if end_video_number else len(self.video_paths) - 1
        self.use_precomputed = use_precomputed

        # Configure Logging
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = log_dir / "preprocess.log"
        self.processed_videos_file = log_dir / "processed_videos.txt"
 
        logging.basicConfig(
            filename=self.log_file,
            filemode="a",
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing dataset preprocessing from video {start_video_number} to {self.end_video_number}")

        # Load Extractor
        self.end_video_number = self.end_video_number or len(self.video_paths)
        self.extractor = FeatureExtractor(use_precomputed=self.use_precomputed)

        # Load processed videos list
        self.processed_videos = self._load_processed_videos()

    def _load_processed_videos(self):
        """Loads the list of already processed videos from a tracking file."""
        if self.processed_videos_file.exists():
            with open(self.processed_videos_file, "r") as f:
                return set(f.read().splitlines())
        return set()

    def _mark_video_as_processed(self, video_name):
        """Marks a video as processed by adding it to the tracking file."""
        with open(self.processed_videos_file, "a") as f:
            f.write(video_name + "\n")

    def process_video(self, video_path, video_index):
        """
        Extracts and saves object depth features for a given video.

        Args:
            video_path (str): Path to the video file from dataset root.
            video_index (int): Index of the video in the dataset.
        """
        video_name = Path(video_path).stem

        # Check if Video is Already Processed
        if str(video_path) in self.processed_videos:
            self.logger.info(f"Skipping {video_path} - Already Processed")
            print(f"Skipping {video_name} - Already Processed")
            return

        try:
            self.logger.info(f"Processing video: {video_path} ({video_index + 1}/{self.end_video_number})")
            start_time = time.time()

            # Extract Object Depth Features
            obj_depth_features = self.extractor.extract_features(video_path, feature_type="object_depth")

            # Save Extracted Features
            save_feature(video_path, obj_depth_features, feature_type="object_depth")

            elapsed_time = time.time() - start_time
            self.logger.info(f"Successfully saved: {video_path} (Time: {elapsed_time:.2f}s)")
            print(f"Features of {video_name} saved successfully.")


            # Mark Video as Processed
            self._mark_video_as_processed(str(video_path))

        except Exception as e:
            self.logger.error(f"Failed to process {video_name} (Error: {str(e)})")
            print(f"Error processing {video_name}: {e}")

    def run(self):
        """Runs the dataset preprocessing pipeline for the specified video range."""
        for i in tqdm(range(self.start_video_number, self.end_video_number), desc=f'Processing dataset'):
            self.process_video(self.video_paths[i], i)

        self.logger.info(f"Preprocessing complete for range {self.start_video_number} to {self.end_video_number}")

 
if __name__ == "__main__":
    mode = 'train'

    start_video = 3000  # Set this for each team member
    end_video = 3449  # Define batch size

    processor = PreprocessDataset(mode=mode, start_video_number=start_video,end_video_number=end_video, use_precomputed=True)
    processor.run()
