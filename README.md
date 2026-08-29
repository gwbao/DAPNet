# Eliminating Subtle Shifts for Real-World Industrial Domain Adaptation

**Note:** This repository provides the official implementation of **DAPNet (Domain-Agnostic Purification Network)**, a method designed to address subtle and fine-grained domain shifts commonly found in professional imaging scenarios, such as X-ray security inspection, remote sensing land-cover segmentation, and medical image analysis.

![](images/model3.png)

In this paper:

  1） We identify a practical UDA challenge termed **``Small shift, Large gap”**, where subtle hardware-induced variations in specialized imaging systems, though barely perceptible in image space, lead to significant feature distribution gaps and degrade cross-domain performance.
  
  2） We propose DAPNet, a domain-agnostic purification framework that addresses subtle domain shift through a synergistic ``expose-then-refine” mechanism: SASM exposes latent domain bias as learnable cross-domain contrastive signals through semantically constrained cross domain feature synthesis, while ACRM leverages the resulting domain feedback to estimate channel-level stability and refine domain-stable semantic representations.
  
 3）Extensive experiments across multiple scenarios, including **X-ray security inspection**, **medical segmentation**, and **remote sensing analysis**, demonstrate that DAPNet outperforms other UDA methods under challenging subtle domain shift scenarios.


---

## Installation

```bash
cd DAPNet
conda create --name dapnet python=3.8 -y
conda activate dapnet

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install mmcv==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

pip install openmim mmengine
mim install "mmcv>=2.0.0rc4"
pip install git+https://github.com/lvis-dataset/lvis-api.git
mim install mmdet>=3.0.0rc6
```

---

## Datasets

We provide loaders for major datasets used in the paper:

* **X-ray Detection:**
  [X-ray Detection Dataset](https://github.com/DIG-Beihang/XrayDetection#eds-endogenous-domain-shift)

* **Remote Sensing Segmentation:**
  Datasets including ISPRS Potsdam, ISPRS Vaihingen can be downloaded [here](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets).

* **Medical Imaging:**
  [Medical Imaging Dataset](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view)
  
* **Traditional cross-domain scenario:**
  [ Cityscapes → Foggy Cityscapes]([https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view](https://github.com/EstrellaXyu/Differential-Alignment-for-DAOD/blob/initial-commit/docs/DATASETS.md))
 You can follow [these instructions](https://github.com/EstrellaXyu/Differential-Alignment-for-DAOD/blob/initial-commit/docs/DATASETS.md) to reproduce the datasets we used.
And all datasets are expected to be organized in the following structure:
```bash
datasets/
    EDS_split/
        domain1/
            annotations/
                test.json
                train.json
            test/
                images/
                labels/
            train/
                ...
        domain2/
            ...
        domain3/
            ...
    Fundus/
        Domain1/
            test/
                image/
                mask/
            train/
                image/
                mask/
        Domain2/
            ...
        Domain3/
            ...
        Domain4/
            ...
    ISPRS/
        potsdam_tiles/
            test/
                images/
                labels/
            train/
                ...
            test.txt
            train.txt
        vaihingen_tiles/
            ...
    cityscapes/
        leftImg8bit/
        leftImg8bit_foggy/
        annotations/
            cityscapes_train_instances.json
            ...
```
After organizing the dataset, you can preceed to configure the corresponding dictionary paths in the dataset.py like:
```bash
# EDS 
register_coco_instances("eds_domain1_train", {},   "",   "")
register_coco_instances("eds_domain2_train", {},     "",     ")
register_coco_instances("eds_domain3_train", {},     "",   "")

register_coco_instances("eds_domain1_test", {},   "",   "")
register_coco_instances("eds_domain2_test", {},     "",   "")
register_coco_instances("eds_domain3_test", {},    "",   "")
```
---

## Training & Test

### Train DAPNet

```bash
python tools/train.py \
    configs/uda/potsdam_vaihingen_stage1.py \
    --work-dir output/xxx \
    --seed 1337
python tools/train.py \
    configs/uda/potsdam_vaihingen_stage2.py \
    --work-dir output/xxx \
    --seed 1337
```

### Test

```bash
python tools/test.py \
    configs/uda/test_potsdam_vaihingen.py \
    output/xxx.pth \
    --eval mIoU mF1 \
    --show-dir output/xx
```
---

## Model zoo

Please download the official pre-trained weights from https://github.com/Sudhandar/ResNet-50-model, https://github.com/hitachinsk/SAMed and [https://github.com/hitachinsk/SAMed](https://github.com/HUMMMZ/MRU-Net). All baseline models in our experiments start with these weights. 
In addition, we provide the DAPNet model weights for X-ray security inspection, medical segmentation, and remote sensing analysis at https://pan.baidu.com/s/14oPAaGXBcfMOAl09Gi1YXw(dapn) for further evaluation.


---

## Citation

```bibtex
@inproceedings{he2025differential,
  title={Differential Alignment for Domain Adaptive Object Detection},
  author={He, Xinyu and Li, Xinhui and Guo, Xiaojie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={16},
  pages={17150--17158},
  year={2025}
}

@inproceedings{tao2022exploring,
  title={Exploring endogenous shift for cross-domain detection: A large-scale benchmark and perturbation suppression network},
  author={Tao, Renshuai and Li, Hainan and Wang, Tianbo and Wei, Yanlu and Ding, Yifu and Jin, Bowei and Zhi, Hongping and Liu, Xianglong and Liu, Aishan},
  booktitle={2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR)},
  pages={21157--21167},
  year={2022},
  organization={IEEE}
}

```
