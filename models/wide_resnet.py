"""
The contents of this file were copied (and slightly modified) from:
https://github.com/yoshitomo-matsubara/torchdistill/blob/main/torchdistill/models/classification/wide_resnet.py
"""

from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor

"""
Refactored https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
for CIFAR datasets, referring to https://github.com/szagoruyko/wide-residual-networks
"""

ROOT_URL = 'https://github.com/yoshitomo-matsubara/torchdistill/releases/download'
MODEL_URL_DICT = {
    'cifar10-wide_resnet40_4': ROOT_URL + '/v0.1.1/cifar10-wide_resnet40_4.pt',
    'cifar10-wide_resnet28_10': ROOT_URL + '/v0.1.1/cifar10-wide_resnet28_10.pt',
    'cifar10-wide_resnet16_8': ROOT_URL + '/v0.1.1/cifar10-wide_resnet16_8.pt',
    'cifar100-wide_resnet40_4': ROOT_URL + '/v0.1.1/cifar100-wide_resnet40_4.pt',
    'cifar100-wide_resnet28_10': ROOT_URL + '/v0.1.1/cifar100-wide_resnet28_10.pt',
    'cifar100-wide_resnet16_8': ROOT_URL + '/v0.1.1/cifar100-wide_resnet16_8.pt'
}


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideBasicBlockV2(WideBasicBlock):
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, depth, k, dropout_p, num_classes, norm_layer=None):
        super().__init__()
        n = (depth - 4) / 6
        stage_sizes = [16, 16 * k, 32 * k, 64 * k]
        in_planes = 16
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, stage_sizes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self.make_wide_layer(in_planes, stage_sizes[1], n, dropout_p, 1)
        self.layer2 = self.make_wide_layer(stage_sizes[1], stage_sizes[2], n, dropout_p, 2)
        self.layer3 = self.make_wide_layer(stage_sizes[2], stage_sizes[3], n, dropout_p, 2)
        self.bn1 = norm_layer(stage_sizes[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_sizes[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_wide_layer(in_planes: int,
                        out_planes: int,
                        num_blocks: int,
                        dropout_rate: float,
                        stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        for stride in strides:
            layers.append(WideBasicBlockV2(in_planes, out_planes, dropout_rate, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def wide_resnet(
        depth: int,
        k: int,
        dropout_p: float,
        num_classes: int,
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> WideResNet:
    assert (depth - 4) % 6 == 0, 'depth of Wide ResNet (WRN) should be 6n + 4'
    model = WideResNet(depth, k, dropout_p, num_classes, **kwargs)
    model_key = 'cifar{}-wide_resnet{}_{}'.format(num_classes, depth, k)
    if pretrained and model_key in MODEL_URL_DICT:
        state_dict = torch.hub.load_state_dict_from_url(MODEL_URL_DICT[model_key], progress=progress)
        model.load_state_dict(state_dict)
    return model


def wide_resnet40_4(dropout_p=0.3, num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> WideResNet:
    r"""WRN-40-4 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        dropout_p (float): p in Dropout
        num_classes (int): 10 and 100 for CIFAR-10 and CIFAR-100, respectively
        pretrained (bool): If True, returns a model pre-trained on CIFAR-10/100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return wide_resnet(40, 4, dropout_p, num_classes, pretrained, progress, **kwargs)


def wide_resnet28_10(dropout_p=0.3, num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> WideResNet:
    r"""WRN-28-10 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        dropout_p: p in Dropout
        num_classes (int): 10 and 100 for CIFAR-10 and CIFAR-100, respectively
        pretrained (bool): If True, returns a model pre-trained on CIFAR-10/100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return wide_resnet(28, 10, dropout_p, num_classes, pretrained, progress, **kwargs)


def wide_resnet16_8(dropout_p=0.3, num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> WideResNet:
    r"""WRN-16-8 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        dropout_p: p in Dropout
        num_classes (int): 10 and 100 for CIFAR-10 and CIFAR-100, respectively
        pretrained (bool): If True, returns a model pre-trained on CIFAR-10/100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return wide_resnet(16, 8, dropout_p, num_classes, pretrained, progress, **kwargs)


if __name__ == '__main__':
    from torchinfo import summary

    wresnet = wide_resnet28_10(num_classes=100)
    summary(wresnet, (1, 3, 32, 32))
