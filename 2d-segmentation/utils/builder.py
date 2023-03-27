from mmcv.cnn import MODELS 
from mmcv.utils import Registry

SEGMENTORS = Registry('models', parent=MODELS)
def build_segmentor(config):
    return SEGMENTORS.build(
        config, default_args=dict(None, test_cfg=config['test_cfg']))