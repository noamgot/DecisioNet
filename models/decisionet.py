from typing import List, Union, Callable, Tuple, Optional, Any

import torch
import torch.nn as nn

from custom_layers.selection_layers import BinarySelectionLayer
from models.network_in_network import NetworkInNetwork
from models.wide_resnet import WideResNet
from utils.binary_tree import Node

ConfigTuple = Tuple[Union[int, str, Tuple[int, int]], ...]
ConfigList = List[Any]

WRESNET_STAGE_SIZES = {'100_baseline': [[(16, 1)], [(16, 2)], [(16, 2)]],
                       '100_baseline_single_early': [[(16, 1)], [(16, 2), (32, 2)]],
                       '100_baseline_single_late': [[(16, 1), (32, 2)], [(32, 2)]]}
NIN_CFG = {'10_baseline': [((192, 5), (160, 1), (96, 1), 'M', 'D'),
                           ((96, 5), (96, 1), (96, 1), 'A', 'D'),
                           ((48, 3), (48, 1), (10, 1))],
           '10_baseline_slim': [((192, 5), (160, 1)),
                                ((48, 1), 'M', 'D', (96, 5), (96, 1)),
                                ((48, 1), 'A', 'D', (48, 3), (48, 1), (10, 1))],
           '10_baseline_single_early': [((192, 5), (160, 1), (96, 1), 'M', 'D'),
                                        ((96, 5), (96, 1), (96, 1), 'A', 'D', (96, 3), (96, 1), (10, 1))],
           '10_baseline_single_late': [((192, 5), (160, 1), (96, 1), 'M', 'D', (192, 5), (192, 1), (192, 1), 'A', 'D'),
                                       ((96, 3), (96, 1), (10, 1))],
           '10_baseline_gap': [((192, 5), (160, 1), (96, 1), 'M', 'D'),
                               ((96, 5), (96, 1), (96, 1), 'A', 'D'),
                               ((48, 3), (48, 1), (10, 1), 'GAP')],
           '100_baseline': [((192, 5), (160, 1), (96, 1), 'M', 'D'),
                            ((96, 5), (96, 1), (96, 1), 'A', 'D'),
                            ((48, 3), (48, 1), (100, 1))],
           '100_baseline_single_early': [((192, 5), (160, 1), (96, 1), 'M', 'D'),
                                         ((96, 5), (96, 1), (96, 1), 'A', 'D', (96, 3), (96, 1), (100, 1))],
           '100_baseline_single_late': [((192, 5), (160, 1), (96, 1), 'M', 'D', (192, 5), (192, 1), (192, 1), 'A', 'D'),
                                        ((96, 3), (96, 1), (100, 1))],
           '100_baseline_gap': [[(192, 5), (160, 1), (96, 1), 'M', 'D'],
                                [(96, 5), (96, 1), (96, 1), 'A', 'D'],
                                [(48, 3), (48, 1), (100, 1), 'GAP']],
           }


class DecisioNet(nn.Module):

    def __init__(self,
                 config: ConfigList,
                 num_in_channels: int,
                 make_layers_func: Callable[[ConfigTuple, int], Tuple[nn.Module, int]],
                 classes_division: Optional[Node] = None,
                 node_code: Tuple[int, ...] = ()):
        super().__init__()
        num_levels = len(config)
        assert num_levels >= 1
        self.is_leaf = (num_levels == 1)
        curr_level_config = config[0]
        self.features, num_out_channels = make_layers_func(curr_level_config, num_in_channels)
        self.node_code = node_code
        if classes_division is not None:
            self.node_classes = classes_division.value
        self.num_out_channels = num_out_channels

        if not self.is_leaf:
            left_cd = classes_division.left if classes_division is not None else None
            right_cd = classes_division.right if classes_division is not None else None
            cls = self.__class__
            self.left = cls(config[1:], num_out_channels, make_layers_func, left_cd, self.node_code + (0,))
            self.right = cls(config[1:], num_out_channels, make_layers_func, right_cd, self.node_code + (1,))
            self.binary_selection_layer = BinarySelectionLayer(num_out_channels)

    def forward(self, x, **kwargs):
        x = self.features(x)
        if self.is_leaf:
            return x, None
        sigma = self.binary_selection_layer(x, **kwargs)
        x0, s0 = self.left(x, **kwargs)
        x1, s1 = self.right(x, **kwargs)
        sigma_broadcasted = sigma[..., None, None] if x0.ndim == 4 else sigma
        x = (1 - sigma_broadcasted) * x0 + sigma_broadcasted * x1
        if s0 is not None and s1 is not None:
            deeper_level_decisions = torch.stack([s0, s1], dim=-1)
            bs = sigma.size(0)
            sigma_idx = sigma.detach().ge(0.5).long().flatten()
            filtered_decisions = deeper_level_decisions[torch.arange(bs), :, sigma_idx]
            sigma = torch.column_stack([sigma, filtered_decisions])
        return x, sigma


class WideResNetDecisioNetNode(nn.Module):

    def __init__(self,
                 stage_sizes: List[List[Tuple[int, int]]],
                 num_blocks: int,
                 num_in_planes: int,
                 num_classes: int,
                 dropout_rate: float = 0.0,
                 classes_division: Optional[Node] = None,
                 node_code: Tuple[int, ...] = (),
                 ops_count_mode: bool = False):
        super().__init__()
        num_levels = len(stage_sizes)
        assert num_levels >= 1
        self.is_leaf = (num_levels == 1)
        # curr_level_num_out_planes, curr_level_stride = stage_sizes[0]
        all_features = []
        out_planes = None
        for s in stage_sizes[0]:
            out_planes, stride = s
            features = WideResNet.make_wide_layer(num_in_planes, out_planes, num_blocks, dropout_rate, stride)
            num_in_planes = out_planes
            all_features.extend(list(features.children()))
        self.features = nn.Sequential(*all_features)
        self.node_code = node_code
        if classes_division is not None:
            self.node_classes = classes_division.value
        # self.num_out_channels = num_out_planes
        assert out_planes
        if self.is_leaf:
            self.features = WideResNetDecisioNet.make_leaf_tail_layers(self.features, out_planes, num_classes)
        else:
            left_cd = classes_division.left if classes_division is not None else None
            right_cd = classes_division.right if classes_division is not None else None
            cls = self.__class__
            self.left = cls(stage_sizes[1:], num_blocks, out_planes, num_classes,
                            dropout_rate, left_cd, self.node_code + (0,), ops_count_mode)
            if not ops_count_mode:
                self.right = cls(stage_sizes[1:], num_blocks, out_planes, num_classes,
                                 dropout_rate, right_cd, self.node_code + (1,))
            self.binary_selection_layer = BinarySelectionLayer(out_planes)

    def forward(self, x, **kwargs):
        x = self.features(x)
        if self.is_leaf:
            return x, None
        sigma = self.binary_selection_layer(x, **kwargs)
        x0, s0 = self.left(x, **kwargs)
        x1, s1 = self.right(x, **kwargs)
        sigma_broadcasted = sigma[..., None, None] if x0.ndim == 4 else sigma
        x = (1 - sigma_broadcasted) * x0 + sigma_broadcasted * x1
        if s0 is not None and s1 is not None:
            deeper_level_decisions = torch.stack([s0, s1], dim=-1)
            bs = sigma.size(0)
            sigma_idx = sigma.detach().ge(0.5).long().flatten()
            filtered_decisions = deeper_level_decisions[torch.arange(bs), :, sigma_idx]
            sigma = torch.column_stack([sigma, filtered_decisions])
        return x, sigma


class MeasurementDn(nn.Module):
    def __init__(self,
                 config: ConfigList,
                 num_in_channels: int,
                 make_layers_func: Callable[[ConfigTuple, int], Tuple[nn.Module, int]],
                 classes_division: Optional[Node] = None,
                 node_code: Tuple[int, ...] = ()):
        super().__init__()
        num_levels = len(config)
        assert num_levels >= 1
        self.is_leaf = (num_levels == 1)
        curr_level_config = config[0]
        self.features, num_out_channels = make_layers_func(curr_level_config, num_in_channels)
        self.node_code = node_code
        if classes_division is not None:
            self.node_classes = classes_division.value
        self.num_out_channels = num_out_channels

        if not self.is_leaf:
            left_cd = classes_division.left if classes_division is not None else None
            cls = self.__class__
            self.left = cls(config[1:], num_out_channels, make_layers_func, left_cd, self.node_code + (0,))
            self.binary_selection_layer = BinarySelectionLayer(num_out_channels)

    def forward(self, x, **kwargs):
        x = self.features(x)
        if self.is_leaf:
            return x, None
        sigma = self.binary_selection_layer(x, **kwargs)
        x0, _ = self.left(x, **kwargs)
        sigma_broadcasted = sigma[..., None, None] if x0.ndim == 4 else sigma
        x = (1 - sigma_broadcasted) * x0
        return x, None


class WideResNetDecisioNet(nn.Module):

    def __init__(self, depth=28, k=10, num_in_channels=3, num_classes=100, cfg_name='100_baseline'):
        super().__init__()
        num_blocks = int((depth - 4) / 6)
        stage_sizes = self._update_stage_sizes(WRESNET_STAGE_SIZES[cfg_name], k)
        in_planes = 16
        self.conv1 = nn.Conv2d(num_in_channels, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.decisionet = WideResNetDecisioNetNode(stage_sizes, num_blocks, in_planes, num_classes)

    @staticmethod
    def _update_stage_sizes(stage_sizes, k):
        for block in stage_sizes:
            for i in range(len(block)):
                num_out_planes, stride = block[i]
                block[i] = (num_out_planes * k, stride)
        return stage_sizes

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        out, sigmas = self.decisionet(x, **kwargs)
        return out, sigmas

    @staticmethod
    def make_leaf_tail_layers(features: nn.Sequential, out_planes: int, num_classes: int):
        features.add_module('bn', nn.BatchNorm2d(out_planes))
        features.add_module('relu', nn.ReLU(inplace=True))
        features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
        features.add_module('flatten', nn.Flatten())
        features.add_module('fc', nn.Linear(out_planes, num_classes))
        return features


class WideResNetDecisioNetNodeMeasurement(WideResNetDecisioNetNode):

    def __init__(self, stage_sizes: List[List[Tuple[int, int]]], num_blocks: int, num_in_planes: int,
                 num_classes: int, dropout_rate: float = 0.0, classes_division: Optional[Node] = None,
                 node_code: Tuple[int, ...] = (), ops_count_mode: bool = False):
        super().__init__(stage_sizes, num_blocks, num_in_planes, num_classes, dropout_rate, classes_division, node_code,
                         ops_count_mode=True)

    def forward(self, x, **kwargs):
        x = self.features(x)
        if self.is_leaf:
            return x, None
        sigma = self.binary_selection_layer(x, **kwargs)
        x0, _ = self.left(x, **kwargs)
        sigma_broadcasted = sigma[..., None, None] if x0.ndim == 4 else sigma
        x = (1 - sigma_broadcasted) * x0
        return x, None


class WideResNetDecisioNetMeasurement(WideResNetDecisioNet):
    def __init__(self, depth=28, k=10, num_in_channels=3, num_classes=100, cfg_name='100_baseline'):
        nn.Module.__init__(self)
        num_blocks = int((depth - 4) / 6)
        stage_sizes = self._update_stage_sizes(WRESNET_STAGE_SIZES[cfg_name], k)
        in_planes = 16
        self.conv1 = nn.Conv2d(num_in_channels, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.decisionet = WideResNetDecisioNetNodeMeasurement(stage_sizes, num_blocks, in_planes, num_classes)


class NetworkInNetworkDecisioNet(nn.Module):
    def __init__(self, cfg_name='10_baseline', dropout=True,
                 classes_division: Optional[Node] = None, decisionet_cls=None,
                 num_in_channels=3):
        super().__init__()
        if decisionet_cls is None:
            decisionet_cls = DecisioNet
        config = NIN_CFG[cfg_name]
        if not dropout:
            config = [x for x in config if x != 'D']
        # config[-1][-1] = (num_classes, 1)
        print("NetworkInNetworkDecisioNet init - Using the following config:")
        print(config)
        self.decisionet = decisionet_cls(config, num_in_channels, NetworkInNetwork.make_layers_by_config,
                                         classes_division)
        self.classifier = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, **kwargs):
        features_out, sigmas = self.decisionet(x, **kwargs)
        out = self.classifier(features_out)
        out = torch.flatten(out, 1)
        return out, sigmas


if __name__ == '__main__':
    from torchinfo import summary

    # model = WideResNetDecisioNet(cfg_name='100_baseline_single_late')
    model = WideResNetDecisioNetMeasurement(cfg_name='100_baseline_single_early')
    # out, sigmas = model(images)
    summary(model, (1, 3, 32, 32))
