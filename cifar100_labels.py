"""
CIFAR-100 Superclass Mapping

CIFAR-100 has 100 fine-grained classes grouped into 20 coarse superclasses.
"""

# Mapping from fine label (0-99) to coarse label (0-19)
CIFAR100_FINE_TO_COARSE = {
    # aquatic mammals
    4: 0, 30: 0, 55: 0, 72: 0, 95: 0,
    # fish
    1: 1, 32: 1, 67: 1, 73: 1, 91: 1,
    # flowers
    54: 2, 62: 2, 70: 2, 82: 2, 92: 2,
    # food containers
    9: 3, 10: 3, 16: 3, 28: 3, 61: 3,
    # fruit and vegetables
    0: 4, 51: 4, 53: 4, 57: 4, 83: 4,
    # household electrical devices
    22: 5, 39: 5, 40: 5, 86: 5, 87: 5,
    # household furniture
    5: 6, 20: 6, 25: 6, 84: 6, 94: 6,
    # insects
    6: 7, 7: 7, 14: 7, 18: 7, 24: 7,
    # large carnivores
    3: 8, 42: 8, 43: 8, 88: 8, 97: 8,
    # large man-made outdoor things
    12: 9, 17: 9, 37: 9, 68: 9, 76: 9,
    # large natural outdoor scenes
    23: 10, 33: 10, 49: 10, 60: 10, 71: 10,
    # large omnivores and herbivores
    15: 11, 19: 11, 21: 11, 31: 11, 38: 11,
    # medium-sized mammals
    34: 12, 63: 12, 64: 12, 66: 12, 75: 12,
    # non-insect invertebrates
    26: 13, 45: 13, 77: 13, 79: 13, 99: 13,
    # people
    2: 14, 11: 14, 35: 14, 46: 14, 98: 14,
    # reptiles
    27: 15, 29: 15, 44: 15, 78: 15, 93: 15,
    # small mammals
    36: 16, 50: 16, 65: 16, 74: 16, 80: 16,
    # trees
    47: 17, 52: 17, 56: 17, 59: 17, 96: 17,
    # vehicles 1
    8: 18, 13: 18, 48: 18, 58: 18, 90: 18,
    # vehicles 2
    41: 19, 69: 19, 81: 19, 85: 19, 89: 19,
}

# Coarse class names
COARSE_LABELS = [
    'aquatic mammals',
    'fish',
    'flowers',
    'food containers',
    'fruit and vegetables',
    'household electrical devices',
    'household furniture',
    'insects',
    'large carnivores',
    'large man-made outdoor things',
    'large natural outdoor scenes',
    'large omnivores and herbivores',
    'medium-sized mammals',
    'non-insect invertebrates',
    'people',
    'reptiles',
    'small mammals',
    'trees',
    'vehicles 1',
    'vehicles 2',
]

# Fine class names
FINE_LABELS = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


def get_coarse_label(fine_label):
    """Convert fine label to coarse label."""
    return CIFAR100_FINE_TO_COARSE[fine_label]
