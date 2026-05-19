# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional OEM and FLAIR datasets

from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset

from .oem import OEMDataset
from .oem_uda import OEM_UDADataset
from .flair import FLAIRDataset
from .flair_uda import FLAIR_UDADataset
from .whu import WHUDataset
from .inria import InriaDataset
from .massachusetts import MassBuildingsDataset
from .isprs_dataset import ISPRSDataset

__all__ = [
    'CustomDataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'FLAIRDataset',
    'FLAIR_UDADataset',
    'OEMDataset',
    'OEM_UDADataset',
    'WHUDataset',
    'InriaDataset',
    'MassBuildingsDataset','ISPRSDataset',]
