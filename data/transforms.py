from typing import List, Any

from torchvision import transforms as transforms

CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)

CIFAR100_MEAN = (0.50707516, 0.48654887, 0.44091784)
CIFAR100_STD = (0.26733429, 0.25643846, 0.27615047)

FashionMNIST_MEAN = (0.2860406,)
FashionMNIST_STD = (0.35302424,)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATA_SETS_MEAN = {'CIFAR10': CIFAR10_MEAN, 'CIFAR100': CIFAR100_MEAN,
                  'ImageNet': IMAGENET_MEAN, 'FashionMNIST': FashionMNIST_MEAN,
                  'default': (0.0, 0.0, 0.0)}
DATA_SETS_STD = {'CIFAR10': CIFAR10_STD, 'CIFAR100': CIFAR100_STD,
                 'ImageNet': IMAGENET_STD, 'FashionMNIST': FashionMNIST_STD,
                 'default': (1.0, 1.0, 1.0)}
DATA_SETS_CROP_SIZE = {'CIFAR10': 32, 'CIFAR100': 32, 'FashionMNIST': 28,  'default': None}


class BasicTransforms:
    def __init__(self, dataset_name: str = 'CIFAR10', augment=True, normalize=True, padding_mode='constant'):
        ds_mean = self.get_val_from_dataset_dict(dataset_name, DATA_SETS_MEAN, 'DATA_SETS_MEAN')
        ds_std = self.get_val_from_dataset_dict(dataset_name, DATA_SETS_STD, 'DATA_SETS_STD')
        crop_size = self.get_val_from_dataset_dict(dataset_name, DATA_SETS_CROP_SIZE, 'DATA_SETS_CROP_SIZE')

        if dataset_name == 'ImageNet':
            data_augmentation_transforms = []
        else:
            data_augmentation_transforms = [
                transforms.RandomCrop(crop_size, padding=crop_size // 8, padding_mode=padding_mode),
                transforms.RandomHorizontalFlip(),
            ]
        common_transforms: List[Any] = [transforms.ToTensor()]
        if dataset_name == 'Food101':
            common_transforms = [transforms.Resize((crop_size, crop_size))] + common_transforms
        if normalize:
            common_transforms.append(transforms.Normalize(ds_mean, ds_std))
        train_transforms = common_transforms
        if augment:
            print("Performing data augmentation!")
            train_transforms.extend(data_augmentation_transforms)
        # train_transforms.extend(common_transforms)
        self.transform_train = transforms.Compose(train_transforms)
        self.transform_validation = transforms.Compose(common_transforms)
        self.transform_test = transforms.Compose(common_transforms)

    @staticmethod
    def get_val_from_dataset_dict(dataset_name: str, dataset_dict: dict, dict_str: str):
        if dataset_name not in dataset_dict:
            print(f"Warning: {dataset_name} is not found in {dict_str} dict - "
                  f"using default value instead: {dataset_dict['default']}")
            return dataset_dict['default']
        return dataset_dict[dataset_name]


class ImageNetTransforms:
    def __init__(self, augment=True, normalize=True, color_jitter=False):

        common_transforms: List[Any] = [transforms.ToTensor()]
        val_transforms = [transforms.Resize(256), transforms.CenterCrop(224)]
        if normalize:
            common_transforms.append(transforms.Normalize(DATA_SETS_MEAN['ImageNet'], DATA_SETS_STD['ImageNet']))
        if augment:
            print("Performing data augmentation!")
            data_augmentation_transforms = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
            if color_jitter:
                data_augmentation_transforms.append(transforms.ColorJitter())
            train_transforms = data_augmentation_transforms + common_transforms
        else:
            train_transforms = val_transforms + common_transforms

        val_test_transforms = val_transforms + common_transforms

        self.transform_train = transforms.Compose(train_transforms)
        self.transform_validation = transforms.Compose(val_test_transforms)
        self.transform_test = transforms.Compose(val_test_transforms)
