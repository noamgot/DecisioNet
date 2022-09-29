"""
This file contains the unused models.
These models were used during our research but are not used in the results that appear in our paper.
"""
from typing import Callable, Tuple, Optional

import torch
import torch.nn as nn

from custom_layers.selection_layers import HardBinaryClassificationLayer
from models import DecisioNet, ConfigList, ConfigTuple, NetworkInNetworkDecisioNet
from utils.binary_tree import Node


class DecisioNetV2(DecisioNet):

    def forward(self, x, **kwargs):
        x = self.features(x)
        if self.is_leaf:
            return x, None
        sigma = self.binary_selection_layer(x, **kwargs)
        x0, s0 = self.left(x, **kwargs)
        x1, s1 = self.right(x, **kwargs)
        sigma_broadcasted = sigma[..., None, None] if x0.ndim == 4 else sigma
        x = torch.cat([(1 - sigma_broadcasted) * x0, sigma_broadcasted * x1], dim=1)
        if s0 is not None and s1 is not None:
            deeper_level_decisions = torch.stack([s0, s1], dim=-1)
            bs = sigma.size(0)
            sigma_idx = sigma.detach().ge(0.5).long().flatten()
            filtered_decisions = deeper_level_decisions[torch.arange(bs), :, sigma_idx]
            sigma = torch.column_stack([sigma, filtered_decisions])
        return x, sigma


class DecisioNetV2New(DecisioNetV2):

    def __init__(self, config: ConfigList, num_in_channels: int,
                 make_layers_func: Callable[[ConfigTuple, int], Tuple[nn.Module, int]],
                 classes_division: Optional[Node] = None, node_code: Tuple[int, ...] = ()):
        super().__init__(config, num_in_channels, make_layers_func, classes_division, node_code)
        if not self.is_leaf:
            self.binary_selection_layer = HardBinaryClassificationLayer(self.num_out_channels, add_noise=False)


class DecisioNetV3(DecisioNet):

    def __init__(self, config: ConfigList, num_in_channels: int,
                 make_layers_func: Callable[[ConfigTuple, int], Tuple[nn.Module, int]],
                 classes_division: Optional[Node] = None, node_code: Tuple[int, ...] = ()):
        assert classes_division is not None
        super().__init__(config, num_in_channels, make_layers_func, classes_division, node_code)

    def forward(self, x, **kwargs):
        x = self.features(x)
        if self.is_leaf:
            b, _, h, w = x.size()
            o = torch.zeros((b, self.num_out_channels, h, w), device=x.device)
            o[:, self.node_classes] = x
            return o, None
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


class NetworkInNetworkDecisioNetV2(NetworkInNetworkDecisioNet):

    def __init__(self, cfg_name='CIFAR10_baseline', dropout=True,
                 classes_division: Optional[Node] = None, decisionet_cls=DecisioNetV2):
        super().__init__(cfg_name, dropout, classes_division, decisionet_cls=decisionet_cls)
        delattr(self, 'classifier')
        self._modify_leaves_outputs()

    def forward(self, x, **kwargs):
        out, sigmas = self.decisionet(x, **kwargs)
        out = torch.flatten(out, 1)
        return out, sigmas

    def _modify_leaves_outputs(self):
        for m in self.decisionet.modules():
            if isinstance(m, DecisioNet) and m.is_leaf:
                if not hasattr(m, 'node_classes'):
                    return
                i = 0
                for l in m.features[::-1]:
                    i -= 1
                    if isinstance(l, nn.Conv2d):
                        num_out_channels = len(m.node_classes)
                        new_conv = nn.Conv2d(l.in_channels, num_out_channels, l.kernel_size, l.stride, l.padding)
                        m.features[i] = new_conv
                        break


class NetworkInNetworkDecisioNetV3(NetworkInNetworkDecisioNet):

    def __init__(self, cfg_name='CIFAR10_baseline', num_classes=10, dropout=True,
                 classes_division: Optional[Node] = None, decisionet_cls=DecisioNetV2):
        super().__init__(cfg_name, dropout, classes_division, decisionet_cls=decisionet_cls)
        delattr(self, 'classifier')
        self.num_classes = num_classes
        self._modify()

    def forward(self, x, **kwargs):
        out, sigmas = self.decisionet(x, **kwargs)
        out = torch.flatten(out, 1)
        return out, sigmas

    def _modify(self):
        ratios = {}
        for m in self.decisionet.modules():
            if isinstance(m, DecisioNet):
                ratios[m.node_code] = len(m.node_classes) / self.num_classes

        for m in self.decisionet.modules():
            if isinstance(m, DecisioNet):
                ratio = ratios[m.node_code]
                parent_ratio = ratios.get(m.node_code[:-1])
                num_in_channels = None
                for i, l in enumerate(m.features):
                    if isinstance(l, nn.Conv2d):
                        if num_in_channels is None:
                            num_in_channels = round(l.in_channels * parent_ratio)
                        num_out_channels = round(ratio * l.out_channels)
                        new_conv = nn.Conv2d(num_in_channels, num_out_channels, l.kernel_size, l.stride, l.padding)
                        num_in_channels = num_out_channels
                        m.features[i] = new_conv
                if not m.is_leaf:
                    m.binary_selection_layer = m.binary_selection_layer.__class__(num_in_channels)


class NetworkInNetworkDecisioNetV4(NetworkInNetworkDecisioNet):

    def __init__(self, cfg_name='10_baseline', num_classes=10, dropout=True,
                 classes_division: Optional[Node] = None, decisionet_cls=DecisioNet):
        super().__init__(cfg_name, dropout, classes_division, decisionet_cls=decisionet_cls)
        self.num_classes = num_classes
        self._modify()

    def _modify(self):
        ratios = {}
        for m in self.decisionet.modules():
            if isinstance(m, DecisioNet):
                ratios[m.node_code] = len(m.node_classes) / self.num_classes

        for m in self.decisionet.modules():
            if isinstance(m, DecisioNet):
                ratio = ratios[m.node_code]
                parent_ratio = ratios.get(m.node_code[:-1])
                num_in_channels = None
                for i, l in enumerate(m.features):
                    if isinstance(l, nn.Conv2d):
                        if num_in_channels is None:
                            num_in_channels = round(l.in_channels * parent_ratio)
                        num_out_channels = round(ratio * l.out_channels)
                        new_conv = nn.Conv2d(num_in_channels, num_out_channels, l.kernel_size, l.stride, l.padding)
                        num_in_channels = num_out_channels
                        m.features[i] = new_conv
                if m.is_leaf:
                    conv = m.features[-2]
                    m.features[-2] = nn.Conv2d(conv.in_channels, self.num_classes,
                                               conv.kernel_size, conv.stride, conv.padding)
                else:
                    m.binary_selection_layer = m.binary_selection_layer.__class__(num_in_channels)
