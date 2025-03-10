# Runs the full preprocessing pipeline for the dataset
from tqdm import tqdm

from data.datasets.ccd.utils import get_video_paths
from data.preprocessing.extract_features import FeatureExtractor



if __name__ == "__main__":
    mode = 'test'

    paths = get_video_paths(split=mode)
    start_video_number = 0
    end_video_number = len(paths)
    extractor = FeatureExtractor(use_precomputed=True)

    for i in tqdm(range(start_video_number, end_video_number)):
        print(i)
        obj_depth_features = extractor.extract_features(paths[i], feature_type="object_depth")
        print(obj_depth_features.shape)
        # save
