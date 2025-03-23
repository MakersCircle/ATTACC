# Runs preprocessing on a single video for inference

from tqdm import tqdm
from data.datasets.ccd.utils import get_video_paths, load_feature

def check_video_features(split='train', start_idx=0, required_frames=50):
    paths = get_video_paths(split=split)

    print(f"🔍 Checking {len(paths)} videos from split: '{split}'...")
    for i in tqdm(range(start_idx, len(paths))):
        path = paths[i]
        vid = path.stem if hasattr(path, 'stem') else str(path)

        try:
            vgg = load_feature(path, feature_type="VGG")
            if vgg.shape[0] != required_frames:
                print(f"❌ VGG frames issue in: {vid}, shape: {vgg.shape}")
                continue

            object_detection = load_feature(path, feature_type="object_detection")
            if object_detection.shape[0] != required_frames:
                print(f"❌ Object detection frames issue in: {vid}, shape: {object_detection.shape}")
                continue

            object_depth = load_feature(path, feature_type="object_depth")
            if object_depth.shape[0] != required_frames:
                print(f"❌ Object depth frames issue in: {vid}, shape: {object_depth.shape}")
                continue

        except Exception as e:
            print(f"❌ Error loading features for {vid}: {e}")
            continue

    print("✅ Dataset integrity check completed.")

if __name__ == '__main__':
    check_video_features(split='train', start_idx=0)

#  55%|█████▍    | 1965/3600 [02:59<02:26, 11.14it/s]❌ Object depth frames issue in: 002058, shape: (49, 19)
# 100%|█████████▉| 3594/3600 [05:29<00:00, 10.98it/s]❌ Object depth frames issue in: 000615, shape: (49, 19)