# mmseg/models/uda/uda_decorator.py
# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Add img_interval
# - Add upscale_pred flag
# - Add DomainInvariantEnhancer support
# - Add _parse_losses method
# - Add get_domain_losses method
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy
from collections import OrderedDict

import torch
from mmcv.parallel import MMDistributedDataParallel
from mmseg.models import BaseSegmentor, build_segmentor


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module


class UDADecorator(BaseSegmentor):
    """
    Base class for Unsupervised Domain Adaptation decorators.

    This class wraps a segmentation model and provides UDA-specific functionality
    including domain adaptation training and domain invariant feature enhancement.
    """

    def __init__(self, **cfg):
        super(BaseSegmentor, self).__init__()

        self.model = build_segmentor(deepcopy(cfg['model']))
        self.train_cfg = cfg['model']['train_cfg']
        self.test_cfg = cfg['model']['test_cfg']
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.debug_img_interval = self.train_cfg['log_config']['img_interval']

        self.use_domain_enhancer = cfg['model'].get('use_domain_enhancer', False)


    def get_model(self):
        """Get the actual model from MMDistributedDataParallel wrapper."""
        return get_module(self.model)

    def extract_feat(self, img):
        """Extract features from images."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas, upscale_pred=True):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img, img_metas, upscale_pred)

    def get_domain_losses(self):
        """

        Returns:
            domain_logits
        """
        if not self.use_domain_enhancer:
            return None

        model = self.get_model()
        if hasattr(model, 'get_domain_losses'):
            return model.get_domain_losses()

        return None

    def _parse_losses(self, losses):

        log_vars = OrderedDict()

        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for sub_name, sub_value in loss_value.items():
                    full_name = f"{loss_name}.{sub_name}"
                    if isinstance(sub_value, torch.Tensor):
                        log_vars[full_name] = sub_value.mean()
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor, list, or dict of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key.lower())

        log_vars['loss'] = loss

        for loss_name, loss_value in log_vars.items():
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      return_feat=False,
                      **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images (source domain).
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            target_img (Tensor): Target domain images (unlabeled).
            target_img_metas (list[dict]): List of target image info dict.
            return_feat (bool): Whether to return features.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=return_feat)
        return losses

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        return self.get_model().inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        return self.get_model().simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        return self.get_model().aug_test(imgs, img_metas, rescale)

    def forward_dummy(self, img):
        """Dummy forward pass for getting FLOPs/params."""
        return self.get_model().forward_dummy(img)

    def train_step(self, data_batch, optimizer, **kwargs):
        raise NotImplementedError(
            "Subclasses must implement train_step method"
        )

    def val_step(self, data_batch, **kwargs):

        output = self.get_model().forward_test(**data_batch, **kwargs)
        return output

    def show_result(self, img, result, palette=None, win_name='',
                    opacity=0.5, out_file=None):

        return self.get_model().show_result(
            img, result, palette=palette, win_name=win_name,
            opacity=opacity, out_file=out_file
        )