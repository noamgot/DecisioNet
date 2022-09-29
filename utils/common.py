import argparse
import time
from typing import Tuple, Union, List

import numpy as np
import scipy.io
import torch
from matplotlib import pyplot as plt


def timeit(func):
    def wrapped(*args, **kwargs):
        s = time.time()
        rv = func(*args, **kwargs)
        e = time.time()
        print(f"Function {func.__name__} took {e - s} seconds")
        return rv

    return wrapped


def imshow(img: torch.Tensor,
           dataset_mean: Union[torch.Tensor, Tuple],
           dataset_std: Union[torch.Tensor, Tuple],
           title: str = '',
           new_figure=False):
    np_img = unnormalize_image(img, dataset_mean, dataset_std)
    if new_figure:
        plt.figure()
    plt.imshow(np_img)
    plt.xticks([])
    plt.yticks([])
    if title:
        plt.title(title)


def unnormalize_image(img: torch.Tensor,
                      dataset_mean: Union[torch.Tensor, Tuple],
                      dataset_std: Union[torch.Tensor, Tuple]) -> np.ndarray:
    if isinstance(dataset_mean, tuple):
        dataset_mean = torch.tensor(dataset_mean, device=img.device)
    if isinstance(dataset_std, tuple):
        dataset_std = torch.tensor(dataset_std, device=img.device)
    img = img * dataset_std[:, None, None] + dataset_mean[:, None, None]  # un-normalize
    np_img = img.cpu().numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    return np_img
