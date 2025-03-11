# Runs the full preprocessing pipeline for the dataset
from tqdm import tqdm

from data.datasets.ccd.utils import get_video_paths, save_feature
from data.preprocessing.extract_features import FeatureExtractor



if __name__ == "__main__":
    mode = 'test'

    paths = get_video_paths(split=mode)
    start_video_number = 0
    end_video_number = len(paths)
    extractor = FeatureExtractor(use_precomputed=True)

    for i in tqdm(range(start_video_number, end_video_number), desc=f'Processing dataset'):
        obj_depth_features = extractor.extract_features(paths[i], feature_type="object_depth")
        # save
        save_feature(paths[i], obj_depth_features, feature_type="object_depth")
        print(f'Features of {paths[i]} is saved.')
