# mmseg/datasets/massachusetts.py

from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class MassBuildingsDataset(CustomDataset):
    """Massachusetts Buildings Dataset.

    The dataset is from https://www.cs.toronto.edu/~vmnih/data/
    The labels are binary (building vs. background), with pixel values 0 and 255.
    """
    CLASSES = ('background', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]] # 黑白

    def __init__(self, **kwargs):
        super(MassBuildingsDataset, self).__init__(
            img_suffix=".png",
            seg_map_suffix=".png",
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None