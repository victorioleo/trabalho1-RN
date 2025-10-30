"""
Example configurations for training different models.
"""

# Configuration for 20 superclasses model
COARSE_CONFIG = {
    'model_type': 'coarse',
    'backbone': 'mobilenetv3_small_100',
    'epochs': 100,
    'batch_size': 128,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
}

# Configuration for 100 classes model
FINE_CONFIG = {
    'model_type': 'fine',
    'backbone': 'mobilenetv3_small_100',
    'epochs': 100,
    'batch_size': 128,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
}

# Configuration for multi-head model
MULTIHEAD_CONFIG = {
    'model_type': 'multihead',
    'backbone': 'mobilenetv3_small_100',
    'epochs': 100,
    'batch_size': 128,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
}

# Alternative backbone options (all under 10M parameters)
# 'mobilenetv3_small_100' - ~2.5M params
# 'mobilenetv3_large_100' - ~5.5M params
# 'efficientnet_b0' - ~5.3M params
# 'resnet18' - ~11M params (slightly over limit)
# 'resnet34' - ~21M params (over limit)
