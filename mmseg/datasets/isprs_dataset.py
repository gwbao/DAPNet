# mmseg/datasets/massachusetts.py

from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class ISPRSDataset(CustomDataset):
    """Massachusetts Buildings Dataset.

    The dataset is from https://www.cs.toronto.edu/~vmnih/data/
    The labels are binary (building vs. background), with pixel values 0 and 255.
    """
    CLASSES = (
        'impervious surface', 'building', 'low vegetation', 'tree', 'car'
    )

    # 调色板，用于可视化，与您的图例严格对应
    PALETTE = [
        [255, 255, 255],  # 0: impervious surface (白色)
        [0, 0, 255],  # 1: building (蓝色)
        [0, 255, 255],  # 2: low vegetation (青色)
        [0, 255, 0],  # 3: tree (绿色)
        [255, 255, 0]  # 4: car (黄色)
    ]

    def __init__(self, **kwargs):
        super(ISPRSDataset, self).__init__(
            img_suffix=".png",
            seg_map_suffix=".png",
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None