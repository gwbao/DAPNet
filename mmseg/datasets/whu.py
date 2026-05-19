from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class WHUDataset(CustomDataset):
    CLASSES = ('background', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]] # 黑白

    def __init__(self, **kwargs):
        super(WHUDataset, self).__init__(
            img_suffix=".png",
            seg_map_suffix=".png",
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None