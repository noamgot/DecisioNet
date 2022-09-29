import collections
import json
import os
from typing import Tuple, Dict

import pandas as pd


def classes_to_idx(classes_list: collections.Iterable):
    return {cls: idx for idx, cls in enumerate(classes_list)}


DATASET_NAMES = ['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet', 'FashionMNIST', 'SVHN', 'Food101']

INPUT_SIZE = {'CIFAR10': (3, 32, 32),
              'CIFAR100': (3, 32, 32),
              'TinyImageNet': (3, 64, 64),
              'ImageNet': (3, 224, 224),
              'FashionMNIST': (1, 28, 28),
              'Food101': (3, 224, 224),
              'SVHN': (3, 32, 32),
              'USPS': (1, 16, 16)}

NUM_CLASSES = {'CIFAR10': 10, 'CIFAR100': 100, 'ImageNet': 1000, 'TinyImageNet': 200, 'FashionMNIST': 10,
               'Food101': 101, 'SVHN': 10, 'USPS': 10}

FASHION_MNIST_CLASSES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
FASHION_MNIST_CLASSES_TO_IDX = classes_to_idx(FASHION_MNIST_CLASSES)
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

CIFAR10_CLASSES_TO_IDX = classes_to_idx(CIFAR10_CLASSES)

CIFAR100_CLASSES = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                    'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                    'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                    'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
                    'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
                    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

CIFAR100_CLASSES_TO_IDX = classes_to_idx(CIFAR100_CLASSES)

file_path = os.path.dirname(__file__)
with open(os.path.join(file_path, 'imagenet1000_clsidx_to_labels.txt'), 'r') as f:
    IMAGENET_CLASSES = []
    imagenet_classes = eval(f.read())
    for i in range(1000):
        IMAGENET_CLASSES.append(imagenet_classes[i].split(',')[0])

IMAGENET_CLASSES_TO_IDX = classes_to_idx(IMAGENET_CLASSES)

CLASSES_NAMES = {'CIFAR10': CIFAR10_CLASSES, 'CIFAR100': CIFAR100_CLASSES,
                 'FashionMNIST': FASHION_MNIST_CLASSES, 'ImageNet': IMAGENET_CLASSES}

####
# CLUSTERING LABEL MAPS

LABELS_MAP: Dict[str, Dict[int, Tuple[int, ...]]] = {
    'FashionMNIST': {0: (0, 0), 1: (0, 1), 2: (0, 0), 3: (0, 0), 4: (0, 0),
                     5: (1, 1), 6: (0, 0), 7: (1, 0), 8: (0, 0), 9: (1, 0)},
    'CIFAR10': {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 0), 4: (1, 0),
                5: (1, 0), 6: (1, 0), 7: (1, 1), 8: (0, 0), 9: (0, 1)},
    'CIFAR10_WRN': {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 0), 4: (1, 0),
                    5: (1, 0), 6: (1, 0), 7: (1, 1), 8: (0, 0), 9: (0, 1)},
    'CIFAR100': {0: (0, 1), 1: (0, 0), 2: (0, 0), 3: (0, 0), 4: (0, 0), 5: (1, 0), 6: (0, 0), 7: (0, 0), 8: (1, 0),
                 9: (1, 0), 10: (1, 0), 11: (0, 0), 12: (1, 0), 13: (1, 0), 14: (0, 0), 15: (0, 0), 16: (1, 0),
                 17: (1, 0), 18: (0, 0), 19: (0, 0), 20: (1, 0), 21: (0, 0), 22: (1, 0), 23: (1, 1), 24: (0, 0),
                 25: (1, 0), 26: (0, 0), 27: (0, 0), 28: (1, 0), 29: (0, 0), 30: (0, 0), 31: (0, 0), 32: (0, 0),
                 33: (1, 1), 34: (0, 0), 35: (0, 0), 36: (0, 0), 37: (1, 0), 38: (0, 0), 39: (1, 0), 40: (1, 0),
                 41: (1, 0), 42: (0, 0), 43: (0, 0), 44: (0, 0), 45: (0, 0), 46: (0, 0), 47: (1, 1), 48: (1, 0),
                 49: (1, 1), 50: (0, 0), 51: (0, 0), 52: (1, 1), 53: (0, 1), 54: (0, 1), 55: (0, 0), 56: (1, 1),
                 57: (0, 1), 58: (1, 0), 59: (1, 1), 60: (1, 1), 61: (1, 0), 62: (0, 1), 63: (0, 0), 64: (0, 0),
                 65: (0, 0), 66: (0, 0), 67: (0, 0), 68: (1, 0), 69: (1, 0), 70: (0, 1), 71: (1, 1), 72: (0, 0),
                 73: (0, 0), 74: (0, 0), 75: (0, 0), 76: (1, 0), 77: (0, 0), 78: (0, 0), 79: (0, 0), 80: (0, 0),
                 81: (1, 0), 82: (0, 1), 83: (0, 1), 84: (1, 0), 85: (1, 0), 86: (1, 0), 87: (1, 0), 88: (0, 0),
                 89: (1, 0), 90: (1, 0), 91: (0, 0), 92: (0, 1), 93: (0, 0), 94: (1, 0), 95: (0, 0), 96: (1, 1),
                 97: (0, 0), 98: (0, 0), 99: (0, 0)},
    'CIFAR100_WRN': {0: (0, 0), 1: (0, 1), 2: (0, 1), 3: (0, 1), 4: (0, 1), 5: (0, 0), 6: (0, 1), 7: (0, 1), 8: (1, 0),
                     9: (0, 0), 10: (0, 0), 11: (0, 1), 12: (1, 0), 13: (1, 0), 14: (0, 1), 15: (0, 1), 16: (0, 0),
                     17: (1, 0), 18: (0, 1), 19: (0, 1), 20: (0, 0), 21: (0, 1), 22: (0, 0), 23: (1, 1), 24: (0, 1),
                     25: (0, 0), 26: (0, 1), 27: (0, 1), 28: (0, 0), 29: (0, 1), 30: (0, 1), 31: (0, 1), 32: (0, 1),
                     33: (1, 1), 34: (0, 1), 35: (0, 1), 36: (0, 1), 37: (1, 0), 38: (0, 1), 39: (0, 0), 40: (0, 0),
                     41: (1, 0), 42: (0, 1), 43: (0, 1), 44: (0, 1), 45: (0, 1), 46: (0, 1), 47: (1, 1), 48: (1, 0),
                     49: (1, 1), 50: (0, 1), 51: (0, 1), 52: (1, 1), 53: (0, 0), 54: (0, 0), 55: (0, 1), 56: (1, 1),
                     57: (0, 0), 58: (1, 0), 59: (1, 1), 60: (1, 1), 61: (0, 0), 62: (0, 0), 63: (0, 1), 64: (0, 1),
                     65: (0, 1), 66: (0, 1), 67: (0, 1), 68: (1, 1), 69: (1, 0), 70: (0, 0), 71: (1, 1), 72: (0, 1),
                     73: (0, 1), 74: (0, 1), 75: (0, 1), 76: (1, 0), 77: (0, 1), 78: (0, 1), 79: (0, 1), 80: (0, 1),
                     81: (1, 0), 82: (0, 0), 83: (0, 0), 84: (0, 0), 85: (1, 0), 86: (0, 0), 87: (0, 0), 88: (0, 1),
                     89: (1, 0), 90: (1, 0), 91: (0, 1), 92: (0, 0), 93: (0, 1), 94: (0, 0), 95: (0, 1), 96: (1, 1),
                     97: (0, 1), 98: (0, 1), 99: (0, 1)},
    'CIFAR100_ext': {0: (0, 1), 1: (0, 0.5), 2: (0, 0), 3: (0, 0), 4: (0, 0), 5: (1, 0), 6: (0, 0), 7: (0, 0),
                     8: (1, 0), 9: (1, 0), 10: (0.5, 0), 11: (0, 0), 12: (1, 0), 13: (1, 0), 14: (0, 0), 15: (0, 0),
                     16: (1, 0), 17: (1, 0), 18: (0, 0), 19: (0, 0), 20: (1, 0), 21: (0, 0), 22: (0.5, 0), 23: (1, 1),
                     24: (0, 0), 25: (0.5, 0), 26: (0, 0), 27: (0, 0), 28: (1, 0), 29: (0, 0), 30: (0, 0), 31: (0, 0),
                     32: (0, 0), 33: (1, 1), 34: (0, 0), 35: (0, 0), 36: (0, 0), 37: (1, 0), 38: (0, 0), 39: (1, 0),
                     40: (0.5, 0), 41: (1, 0), 42: (0, 0), 43: (0, 0), 44: (0, 0), 45: (0.5, 0), 46: (0, 0), 47: (1, 1),
                     48: (1, 0), 49: (1, 1), 50: (0, 0), 51: (0, 0), 52: (1, 1), 53: (0, 1), 54: (0, 0.5), 55: (0, 0),
                     56: (1, 1), 57: (0, 1), 58: (1, 0), 59: (1, 1), 60: (1, 1), 61: (0.5, 0), 62: (0, 1), 63: (0, 0),
                     64: (0, 0), 65: (0, 0), 66: (0, 0), 67: (0, 0), 68: (1, 0), 69: (0.5, 0), 70: (0, 1), 71: (1, 1),
                     72: (0, 0), 73: (0, 0), 74: (0, 0), 75: (0, 0), 76: (1, 0), 77: (0, 0), 78: (0, 0), 79: (0, 0),
                     80: (0, 0), 81: (1, 0), 82: (0, 1), 83: (0, 1), 84: (1, 0), 85: (1, 0), 86: (1, 0), 87: (1, 0),
                     88: (0, 0), 89: (1, 0), 90: (1, 0), 91: (0, 0), 92: (0, 1), 93: (0, 0), 94: (1, 0), 95: (0, 0),
                     96: (1, 1), 97: (0, 0), 98: (0, 0), 99: (0, 0)},
    'CIFAR100_ext2': {0: (0, 1), 1: (0, 0), 2: (0, 0), 3: (0, 0), 4: (0, 0), 5: (1, 0), 6: (0, 0), 7: (0, 0), 8: (1, 0),
                      9: (1, 0), 10: (0.5, 0.5), 11: (0, 0), 12: (1, 0), 13: (1, 0), 14: (0, 0), 15: (0.5, 0.5),
                      16: (1, 0), 17: (1, 0), 18: (0, 0), 19: (0, 0), 20: (1, 0), 21: (0, 0), 22: (0.5, 0.5),
                      23: (1, 1), 24: (0, 0), 25: (1, 0), 26: (0, 0), 27: (0, 0), 28: (0.5, 0.5), 29: (0, 0),
                      30: (0, 0), 31: (0.5, 0.5), 32: (0.5, 0.5), 33: (0.5, 0.5), 34: (0, 0), 35: (0, 0), 36: (0, 0),
                      37: (1, 0), 38: (0, 0), 39: (0.5, 0.5), 40: (0.5, 0.5), 41: (0.5, 0.5), 42: (0, 0), 43: (0, 0),
                      44: (0, 0), 45: (0, 0), 46: (0, 0), 47: (1, 1), 48: (1, 0), 49: (1, 1), 50: (0, 0), 51: (0, 0),
                      52: (1, 1), 53: (0, 1), 54: (0, 1), 55: (0, 0), 56: (0.5, 0.5), 57: (0, 1), 58: (1, 0),
                      59: (1, 1), 60: (1, 1), 61: (0.5, 0.5), 62: (0, 1), 63: (0, 0), 64: (0, 0), 65: (0, 0),
                      66: (0, 0), 67: (0, 0), 68: (1, 0), 69: (1, 0), 70: (0, 1), 71: (1, 1), 72: (0, 0), 73: (0, 0),
                      74: (0, 0), 75: (0, 0), 76: (1, 0), 77: (0, 0), 78: (0, 0), 79: (0, 0), 80: (0, 0), 81: (1, 0),
                      82: (0, 1), 83: (0, 1), 84: (1, 0), 85: (1, 0), 86: (1, 0), 87: (1, 0), 88: (0, 0),
                      89: (0.5, 0.5), 90: (1, 0), 91: (0, 0), 92: (0, 1), 93: (0, 0), 94: (1, 0), 95: (0, 0),
                      96: (1, 1), 97: (0, 0), 98: (0, 0), 99: (0.5, 0.5)}
}

# ADDING THE CLASS LABELS:
for ds_labels_map in LABELS_MAP.values():
    for cls_idx, labels in ds_labels_map.items():
        ds_labels_map[cls_idx] = (cls_idx,) + labels

if __name__ == '__main__':
    dfs = []
    for ds_name in ['CIFAR10', 'FashionMNIST']:
        labels_map = LABELS_MAP[ds_name]
        ds_class_names = CLASSES_NAMES[ds_name]
        orig_keys = list(labels_map.keys())
        for k in orig_keys:
            labels_map[ds_class_names[k]] = labels_map.pop(k)[1:]
        # noinspection PyTypeChecker
        labels_map = dict(sorted(labels_map.items(), key=lambda x: (x[1][0], x[1][1])))
        df = pd.DataFrame.from_dict(labels_map, orient='index', columns=['1st level cluster', '2nd level cluster'])
        dfs.append(df)
    cdf = pd.concat(dfs)
    print(cdf.to_latex())
