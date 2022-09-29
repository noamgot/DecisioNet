import torch

from data.datasets import BasicDatasets
from data.transforms import BasicTransforms, DATA_SETS_MEAN, DATA_SETS_STD, ImageNetTransforms
from utils.common import imshow
from utils.constants import CLASSES_NAMES


class BasicDataLoaders:
    def __init__(self, datasets: BasicDatasets, use_validation=True, train_bs=64,
                 validation_bs=100, test_bs=100, num_workers=2):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.train_set, batch_size=train_bs, shuffle=True, num_workers=num_workers)
        if use_validation:
            self.validation_loader = torch.utils.data.DataLoader(
                datasets.validation_set, batch_size=validation_bs, shuffle=False, num_workers=num_workers)
        else:
            self.validation_loader = None
        self.test_loader = torch.utils.data.DataLoader(
            datasets.test_set, batch_size=test_bs, shuffle=False, num_workers=num_workers)


if __name__ == '__main__':
    import torchvision
    from matplotlib import pyplot as plt

    dataset_name = 'CIFAR10'
    classes_names = CLASSES_NAMES[dataset_name]
    if dataset_name == 'ImageNet':
        transforms_obj = ImageNetTransforms(augment=True)
    else:
        transforms_obj = BasicTransforms(dataset_name=dataset_name, augment=False)
    dl = BasicDataLoaders(
        BasicDatasets(transforms_obj,
                      use_validation=False,
                      dataset_name=dataset_name),
        use_validation=False, num_workers=0)
    data_iter = iter(dl.train_loader)
    for i in range(10):
        images, labels = next(data_iter)
        img_idx = torch.randint(0, images.size(0), (1,)).item()
        imshow(images[img_idx], DATA_SETS_MEAN[dataset_name], DATA_SETS_STD[dataset_name],
               classes_names[labels[img_idx].item()], new_figure=True)
    images, _ = next(data_iter)
    imshow(torchvision.utils.make_grid(images), DATA_SETS_MEAN[dataset_name], DATA_SETS_STD[dataset_name],
           new_figure=True)
    plt.show()
