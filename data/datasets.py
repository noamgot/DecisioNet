import os
from typing import Tuple, Optional, Any

import numpy as np
import torch
import torchvision.datasets
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, ImageFolder, FashionMNIST
from torchvision.transforms import transforms
from tqdm import tqdm

from data.transforms import BasicTransforms
from utils.common import timeit
from utils.constants import NUM_CLASSES

DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')  # change to the path where your data should be stored
IMAGENET_DIRNAME = 'imagenet'
IMAGE_FOLDER_DIRNAMES = {'ImageNet': IMAGENET_DIRNAME}


class FashionMNISTWrapper(FashionMNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASETS = {'CIFAR10': CIFAR10, 'CIFAR100': CIFAR100, 'ImageNet': ImageFolder, 'FashionMNIST': FashionMNISTWrapper}


class BasicDatasets:
    """
    This class holds training, test, and (optional) validation datasets.
    """

    def __init__(self, transforms_: BasicTransforms, use_validation: bool = True, dataset_name: str = 'CIFAR10'):
        assert dataset_name in DATASETS, f"Invalid dataset: {dataset_name}"
        dataset = DATASETS[dataset_name]

        train_kwargs = self.get_kwargs('train', transforms_, dataset, IMAGE_FOLDER_DIRNAMES.get(dataset_name))
        self.train_set = dataset(**train_kwargs)
        if use_validation or dataset_name == 'ImageNet':
            if dataset == ImageFolder:
                validation_kwargs = self.get_kwargs('validation', transforms_, dataset,
                                                    IMAGE_FOLDER_DIRNAMES.get(dataset_name))
                self.validation_set = dataset(**validation_kwargs)
            else:
                num_images = len(self.train_set)
                split_ratios = [int(0.9 * num_images), int(0.1 * num_images)]
                self.train_set, self.validation_set = torch.utils.data.random_split(self.train_set, split_ratios)
                self.validation_set.dataset.transform = transforms_.transform_validation
        else:
            self.validation_set = None
        if dataset_name == 'ImageNet':
            self.test_set = self.validation_set
        else:
            test_kwargs = self.get_kwargs('test', transforms_, dataset, IMAGE_FOLDER_DIRNAMES.get(dataset_name))
            self.test_set = dataset(**test_kwargs)

    @staticmethod
    def get_kwargs(tvt: str, transforms_: BasicTransforms, dataset, ds_dirname=None):
        assert tvt in ['train', 'validation', 'test']
        tvt_transforms = getattr(transforms_, f'transform_{tvt}')
        kwargs = {"transform": tvt_transforms}
        if dataset in [ImageFolder, ImageNet]:
            if ds_dirname is None:
                raise Exception("Must provide the dataset dirname when using ImageFolder dataset")
            if tvt == 'validation':
                tvt = 'val'
            kwargs['root'] = os.path.join(DATA_ROOT_PATH, f"{ds_dirname}/{tvt}")
        else:
            kwargs['root'] = DATA_ROOT_PATH
            kwargs['download'] = True
            kwargs['train'] = (tvt == 'train')
        return kwargs


class FilteredDatasets(BasicDatasets):
    """
    This class allows filtering classes out of the initialized datasets.
    """

    def __init__(self, transforms_: BasicTransforms, use_validation: bool = True, dataset_name: str = 'CIFAR10',
                 classes_indices: Optional[np.ndarray] = None):
        super().__init__(transforms_, use_validation, dataset_name)
        self.classes_indices = classes_indices
        self.filter_dataset(dataset=self.train_set)
        if use_validation:
            self.filter_dataset(dataset=self.validation_set)
        self.filter_dataset(dataset=self.test_set)

    def filter_dataset(self, dataset):
        if self.classes_indices is None or np.all(self.classes_indices == np.unique(dataset.targets)):
            return
        targets_ = torch.tensor(dataset.targets)
        idx_mask = np.isin(targets_, self.classes_indices)
        dataset.data = dataset.data[idx_mask]
        dataset.targets = targets_[idx_mask].tolist()


class FilteredRelabeledDatasets(FilteredDatasets):
    """
    This class allows filtering and relabeling classes of the initialized datasets.
    """

    def __init__(self, transforms_: BasicTransforms, use_validation: bool = True, dataset_name: str = 'CIFAR10',
                 classes_indices: Optional[np.ndarray] = None, labels_map: Optional[dict] = None):
        super().__init__(transforms_, use_validation, dataset_name, classes_indices)
        if self.classes_indices is None:
            self.classes_indices = np.arange(NUM_CLASSES[dataset_name])
        self.labels_map = labels_map  # key = class_idx, values = new label(s)
        self.relabel_dataset(dataset=self.train_set)
        if use_validation:
            self.relabel_dataset(dataset=self.validation_set)
        self.relabel_dataset(dataset=self.test_set)

    def relabel_dataset(self, dataset):
        if self.labels_map is None:
            return

        assert self.classes_indices is not None and isinstance(self.labels_map, dict)
        if isinstance(dataset.targets, torch.Tensor):
            orig_targets = dataset.targets.clone()
        else:
            orig_targets = torch.tensor(dataset.targets)

        first_labels_map_val = list(self.labels_map.values())[0]
        num_labels = len(first_labels_map_val) if isinstance(first_labels_map_val, tuple) else 1
        relabeled_targets = torch.zeros((len(dataset), num_labels))
        for idx, new_label in self.labels_map.items():
            new_label = torch.tensor(new_label, dtype=relabeled_targets.dtype)
            class_mask = (orig_targets == idx)
            relabeled_targets[class_mask] = new_label
        dataset.targets = relabeled_targets.squeeze().tolist()


@timeit
def get_mean_and_std_gt(dataset, verbose=True, image_format='nhwc'):
    """Compute the mean and std value of dataset."""
    axis = (0, 1, 2) if image_format == 'nhwc' else (0, 2, 3)
    data = dataset.data
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if data.ndim == 3:
        data = data[..., None] if image_format == 'nhwc' else data[:, None, ...]
    mean = data.mean(axis=axis) / 255.
    std = data.std(axis=axis) / 255.
    if verbose:
        print("Mean:", mean)
        print("Std:", std)
    return mean, std


@timeit
def get_mean_and_std(dataset: torchvision.datasets.VisionDataset, batch_size=32, num_workers=0, verbose=True,
                     shuffle=False):
    """Compute the mean and std value of dataset iteratively."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    #####
    num_pixels_per_channel = 0
    sum_ = torch.zeros(3)
    sum_squared = torch.zeros(3)
    for images, _ in tqdm(dataloader):
        b, c, h, w = images.size()
        num_pixels_per_channel += b * h * w
        sum_ += images.sum(dim=(0, 2, 3))
        sum_squared += torch.square(images).sum(dim=(0, 2, 3))
    mean = sum_ / num_pixels_per_channel
    var = (sum_squared / num_pixels_per_channel - (mean ** 2))
    std = torch.sqrt(var)
    mean = mean.numpy()
    std = std.numpy()
    if verbose:
        print("Mean:", mean)
        print("Std:", std)
    return mean, std


if __name__ == '__main__':
    num_workers = len(os.sched_getaffinity(0)) if torch.cuda.is_available() else 0
    print(f"Using {num_workers} workers")
    fashion_mnist_ds = FashionMNIST(root=DATA_ROOT_PATH, train=True, download=True, transform=transforms.ToTensor())
    train_set = fashion_mnist_ds
    mean_gt, std_gt = get_mean_and_std(train_set, num_workers=num_workers, verbose=True)
    # get_mean_and_std_gt(train_set, verbose=True, image_format='nchw')
