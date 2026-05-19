# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support UDA models

import warnings

#from mmcv.cnn import MODELS as MMCV_MODELS
#from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
#from mmcv.utils import Registry


# 兼容不同版本的 mmcv
try:
    from mmcv.cnn import MODELS as MMCV_MODELS
    from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
except ImportError:
    try:
        # 尝试从 mmcv.utils 导入 (mmcv 1.x)
        from mmcv.utils import Registry

        MMCV_MODELS = Registry('mmcv_models')
        MMCV_ATTENTION = Registry('mmcv_attention')
    except ImportError:
        # 尝试从 mmengine 导入 (mmcv 2.x)
        try:
            from mmengine.registry import Registry

            MMCV_MODELS = Registry('mmcv_models')
            MMCV_ATTENTION = Registry('mmcv_attention')
        except ImportError:
            # 如果都失败，创建简单的注册器
            class SimpleRegistry:
                def __init__(self, name, parent=None):
                    self.name = name
                    self.parent = parent
                    self._module_dict = {}

                def register_module(self, name=None, force=False, module=None):
                    def _register(cls):
                        module_name = name if name is not None else cls.__name__
                        self._module_dict[module_name] = cls
                        return cls

                    if module is not None:
                        return _register(module)
                    return _register

                def build(self, cfg, default_args=None):
                    if isinstance(cfg, dict):
                        cfg = cfg.copy()
                        module_type = cfg.pop('type')
                        if default_args is not None:
                            for key, value in default_args.items():
                                cfg.setdefault(key, value)
                        return self._module_dict[module_type](**cfg)
                    else:
                        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')


            MMCV_MODELS = SimpleRegistry('mmcv_models')
            MMCV_ATTENTION = SimpleRegistry('mmcv_attention')
            Registry = SimpleRegistry

try:
    from mmcv.utils import Registry
except ImportError:
    try:
        from mmengine.registry import Registry
    except ImportError:
        pass  # 使用上面定义的 SimpleRegistry

MODELS = Registry('models', parent=MMCV_MODELS)
ATTENTION = Registry('attention', parent=MMCV_ATTENTION)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS
UDA = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_train_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.model.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.model.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    if 'uda' in cfg:
        cfg.uda['model'] = cfg.model
        cfg.uda['max_iters'] = cfg.runner.max_iters
        return UDA.build(
            cfg.uda, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    else:
        return SEGMENTORS.build(
            cfg.model,
            default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
