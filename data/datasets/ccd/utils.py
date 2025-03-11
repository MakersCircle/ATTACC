import yaml
import numpy as np
from pathlib import Path


def load_config():
    """Load the CCD dataset configuration file."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_video_paths(split="train"):
    """Get the list of video file paths for train/test sets."""
    config = load_config()
    txt_file = Path(__file__).parent / config[f"{split}_list"]

    video_paths = []
    with open(txt_file, "r") as f:
        for line in f.readlines():
            video_name = line.strip().split()[0]
            label = "positive" if "positive" in video_name else "negative"
            video_path = Path(config["video_path"][label]) / f"{Path(video_name).stem}.mp4"
            video_paths.append(video_path)

    return video_paths


def get_feature_path(video_path):
    """Get the path to a precomputed feature file."""
    config = load_config()
    video_name = video_path.stem
    label = "positive" if "positive" in str(video_path) else "negative"

    feature_base = Path(config["precomputed_features"]["feature_path"][label])
    feature_path = Path(__file__).parent / feature_base / f"{video_name}.npz"
    print(f"Loading features from {feature_path}")
    return feature_path if feature_path.exists() else None


def load_feature(video_path, feature_type):
    """Load a specific feature (e.g., VGG, depth) from precomputed files of the given video."""
    feature_path = get_feature_path(video_path)
    if feature_path:
        if feature_type == "VGG":
            return np.load(feature_path)['data']
        elif feature_type == "object_detection":
            return np.load(feature_path)['det']
    return None


def save_feature(video_path, feature_dict, feature_type):
    """Save extracted features to `extracted_features/` directory."""
    config = load_config()
    video_name = video_path.stem
    label = "positive" if "positive" in str(video_path) else "negative"

    save_path = Path(__file__).parent / Path(config["extracted_features"][label])
    save_path.mkdir(parents=True, exist_ok=True)

    feature_file = save_path / f"{video_name}_{feature_type}.npy"
    np.save(feature_file, feature_dict)
    print(f"Saved features for {video_name}: {feature_file}")


if __name__ == "__main__":
    # Example usage
    config = load_config()
    print("Dataset Structure:", config["dataset_structure"])

    train_videos = get_video_paths("train")
    print(f"Found {len(train_videos)} training videos.")
    print(train_videos[0])

    sample_video = train_videos[0]
    feature_data = load_feature(sample_video, "vgg")
    if feature_data is not None:
        print(f"Loaded VGG features for {sample_video}")
    else:
        print(f"No precomputed VGG features found for {sample_video}")

    # Example feature saving
    save_feature(sample_video, {"depth": np.array([1, 2, 3])}, feature_type="depth")
