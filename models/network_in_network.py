import torch
import torch.nn as nn

from utils.constants import INPUT_SIZE, NUM_CLASSES


class NetworkInNetwork(nn.Module):
    CONFIG = [(192, 5), (160, 1), (96, 1), 'M', 'D',
              (192, 5), (192, 1), (192, 1), 'A', 'D',
              (192, 3), (192, 1), (10, 1)]

    def __init__(self, num_classes=10, num_in_channels=3, dropout=True, config=None, verbose=True):
        super(NetworkInNetwork, self).__init__()
        if config is None:
            config = self.CONFIG
        if not dropout:
            config = [x for x in config if x != 'D']
        config[-1] = (num_classes, 1)
        if verbose:
            print("NetworkInNetwork init - Using the following config:")
            print(config)
        self.features, _ = self.make_layers_by_config(config, num_in_channels=num_in_channels)
        self.classifier = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x

    @staticmethod
    def make_layers_by_config(cfg, num_in_channels=3):
        layers = []
        in_channels = num_in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]
            elif x == 'D':
                layers += [nn.Dropout(0.5)]
            elif x == 'GAP':
                layers += [nn.AdaptiveAvgPool2d((1, 1))]
            elif x == 'GMP':
                layers += [nn.AdaptiveMaxPool2d((1, 1))]
            else:
                out_channels, kernel_size = x
                padding = kernel_size // 2
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                           nn.ReLU(inplace=True)]
                in_channels = out_channels
        return nn.Sequential(*layers), in_channels


if __name__ == '__main__':
    from torchinfo import summary

    verbose = True
    ds_name = 'CIFAR10'
    c, h, w = INPUT_SIZE[ds_name]
    num_classes = NUM_CLASSES[ds_name]
    nin_baseline = NetworkInNetwork(num_classes=num_classes, num_in_channels=c, verbose=verbose)
    nin_baseline_summary = summary(nin_baseline, (1, c, h, w), verbose=verbose)
