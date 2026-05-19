from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class InriaDataset(CustomDataset):
    CLASSES = ('background', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]] # 黑白

    def __init__(self, **kwargs):
        super(InriaDataset, self).__init__(
            img_suffix=".tif",
            seg_map_suffix=".tif",
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None