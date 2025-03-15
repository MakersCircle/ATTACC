# Currently this file is specifically written for ccd dataset. Should be updated to generalize for all datasets

from torch.utils.data import Dataset




class VideoDataset(Dataset):
    def __init__(self, is_train=True):
        pass


    def __len__(self):
        pass


    def _get_tta(self, bin_labels):
        pass


    def __getitem__(self, index):

        # return object_detections, object_depths, object_features, frame_features, bin_labels, tta, cls
        pass
