"""
This file contains old sigma-loss trainers that are not used anymore.
"""
import random
from typing import Union, TextIO

import numpy as np
import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt

from custom_layers.losses import WeightedMSELoss
from data.datasets import FilteredRelabeledDatasets
from data.transforms import DATA_SETS_MEAN, DATA_SETS_STD
from models.decisionet import NetworkInNetworkDecisioNet, NIN_CFG, WideResNetDecisioNet
from models.decisionet_unused_models import NetworkInNetworkDecisioNetV2, DecisioNetV3, DecisioNetV2New, \
    NetworkInNetworkDecisioNetV3, DecisioNetV2, NetworkInNetworkDecisioNetV4
from trainers.baseline_trainers import NetworkInNetworkTrainer
from trainers.basic_trainer import BasicTrainer
from trainers.decisionet_trainers import NetworkInNetworkDecisioNetTrainer
from utils.binary_tree import Node
from utils.common import unnormalize_image
from utils.constants import LABELS_MAP, CLASSES_NAMES, INPUT_SIZE, NUM_CLASSES
from utils.metrics_tracker import SigmaLossMetricsTracker


class NetworkInNetworkDecisioNetTrainerSigmaLossV3(NetworkInNetworkDecisioNetTrainer):

    def init_early_stopping(self):
        self.early_stopping_params = {'mode': 'min', 'patience': 50, 'verbose': True}
        return BasicTrainer.init_early_stopping(self)

    def early_stopping_step(self, train_acc=None, train_loss=None, test_acc=None, test_loss=None):
        return BasicTrainer.early_stopping_step(self, train_acc, train_loss, test_acc, test_loss)

    def _feed_forward(self, inputs, targets):
        cls_targets, *sigma_targets = targets
        sigma_targets = torch.column_stack(sigma_targets)
        binarize = self.always_binarize or None  # or random.random() > 0.5
        orig_outputs, sigmas = self.model(inputs, binarize=binarize)
        outputs = orig_outputs.clone()
        outputs[:, self.classes_output_mapping] = orig_outputs
        cls_loss = self.cls_criterion(outputs, cls_targets.long())
        sigma_loss = self.sigma_criterion(sigmas, sigma_targets)
        combined_loss = cls_loss + self.beta * sigma_loss
        self.metrics_tracker.update(cls_loss, sigma_loss, combined_loss, outputs, cls_targets, sigmas, sigma_targets)
        return outputs, combined_loss

    def _init_model(self):
        self.classes_division_tree = self._init_classes_division_tree()
        self.classes_output_mapping = self._calc_classes_output_mapping()
        model = NetworkInNetworkDecisioNetV2(cfg_name='CIFAR100_baseline_v2',
                                             classes_division=self.classes_division_tree)
        # model = NetworkInNetworkDecisioNetV2(cfg_name='CIFAR100_baseline_balanced')
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.normal_(0, 0.0)
        return model

    def _init_classes_division_tree(self):
        labels_map = dict(LABELS_MAP[self.dataset_name])
        for k, v in labels_map.items():
            labels_map[k] = v[1:]
        tree = Node.init_from_dict(labels_map)
        return tree

    # noinspection PyTypeChecker
    def _calc_classes_output_mapping(self):
        num_leaves = len(self.classes_division_tree.leaves)
        indices = [None] * num_leaves
        level_order = self.classes_division_tree.level_order_encode()
        for node, code in level_order:
            if node.is_leaf():
                idx = int('0b' + ''.join(map(str, code)), 2)
                assert indices[idx] is None
                indices[idx] = torch.tensor(node.value)
        indices = torch.cat(indices)
        return indices

    def get_sample_predictions(self, num_images=24):
        output_predictions = []
        output_images = []
        test_set = self.datasets.test_set
        num_images = min(num_images, len(test_set))
        data_loader = torch.utils.data.DataLoader(test_set, batch_size=num_images, shuffle=True, num_workers=0)
        data_iter = iter(data_loader)
        ds_mean = DATA_SETS_MEAN[self.dataset_name]
        ds_std = DATA_SETS_STD[self.dataset_name]
        self.model.eval()
        with torch.no_grad():
            images, labels = next(data_iter)
            images, labels = self.input_and_targets_to_device(images, labels)
            orig_predictions, _ = self.model(images)
            predictions = orig_predictions.clone()
            predictions[:, self.classes_output_mapping] = orig_predictions
            _, predictions = predictions.max(1)
            for i in range(num_images):
                img = images[i]
                pred = predictions[i]
                pred = CLASSES_NAMES[self.dataset_name][int(pred)]
                np_img = unnormalize_image(img, ds_mean, ds_std)
                output_images.append(np_img)
                output_predictions.append(pred)
        sample_predictions = [wandb.Image(image, caption=prediction)
                              for (image, prediction) in zip(output_images, output_predictions)]
        return sample_predictions

    # noinspection PyTypeChecker
    def correlation_analysis(self):
        activations_dict = {}
        self.register_hooks(activations_dict)
        NetworkInNetworkTrainer.evaluate(self)
        # norm_acc, _ = self._test_single_epoch(0)
        targets = torch.tensor(self.datasets.test_set.targets)
        predictions = activations_dict['predictions']
        for i in range(predictions.size(0)):
            predictions[i] = self.classes_output_mapping[predictions[i].long()]
        sigmas = activations_dict['sigma']
        cls_targets = targets[:, 0]
        sigma_targets = targets[:, 1:]

        encoding = {0: 'both wrong', 1: 'first correct', 2: 'second correct', 3: 'both correct'}
        all_accuracies = np.zeros((5, self.num_classes))
        for cls in self.classes_indices:
            acc_list = []
            cls_idx = torch.where(cls_targets == cls)[0]
            num_samples = cls_idx.size(0)
            cls_acc = torch.sum(predictions[cls_idx] == cls).item() / num_samples
            acc_list.append(cls_acc)
            sigma_diffs = (sigmas[cls_idx] == sigma_targets[cls_idx])
            encoded_results = torch.sum(sigma_diffs * torch.tensor([1., 2.]), dim=1)
            for code, s in encoding.items():
                code_acc = torch.sum(encoded_results == code).item() / num_samples
                acc_list.append(code_acc)
            all_accuracies[:, cls] = np.array(acc_list)
            # plt.bar(list(encoding.values()), results)
        # plt.figure()
        # for i, t in enumerate(['acc'] + list(encoding.values())):
        #     plt.scatter(np.arange(self.num_classes), all_accuracies[i], label=t)
        # plt.legend()
        # plt.show()
        pearson_coeff = np.corrcoef(all_accuracies)
        print(pearson_coeff[0, 1:])


class NetworkInNetworkDecisioNetTrainerSigmaLossV4(NetworkInNetworkDecisioNetTrainerSigmaLossV3):

    def __init__(self, classes_indices=None):
        super().__init__()
        sigma_weights = self._init_sigma_weights()
        self.sigma_criterion = WeightedMSELoss(sigma_weights, ignore_value=0.5)

    def _init_model(self):
        self.classes_division_tree = self._init_classes_division_tree()
        model = NetworkInNetworkDecisioNetV2(cfg_name='CIFAR100_baseline_v2',
                                             classes_division=self.classes_division_tree, decisionet_cls=DecisioNetV3)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.normal_(0, 0.0)
        return model

    def _init_classes_division_tree(self):
        labels_map = dict(LABELS_MAP[self.dataset_name + '_ext2'])
        for k, v in labels_map.items():
            labels_map[k] = v[1:]
        tree = Node.init_from_dict_extended(labels_map)
        # tree = Node.init_from_dict(labels_map)
        return tree

    def init_data_sets(self):
        labels_map = LABELS_MAP[self.dataset_name + '_ext2']
        return FilteredRelabeledDatasets(self.transforms, use_validation=self.use_validation,
                                         classes_indices=self.classes_indices,
                                         labels_map=labels_map,
                                         dataset_name=self.dataset_name)

    def _feed_forward(self, inputs, targets):
        return NetworkInNetworkDecisioNetTrainer._feed_forward(self, inputs, targets)

    def evaluate(self):
        return NetworkInNetworkTrainer.evaluate(self)


class NinDnPretrained(NetworkInNetworkDecisioNetTrainerSigmaLossV3):

    def __init__(self, classes_indices=None):
        super().__init__()
        self.sigma_criterion = nn.BCELoss()

    def evaluate(self):
        return BasicTrainer.evaluate(self)

    def init_parser(self):
        parser = super().init_parser()
        for action in parser._actions:
            if action.dest == 'weights_path':
                action.nargs = '+'
                break
        return parser

    def _init_model(self):
        self.classes_division_tree = self._init_classes_division_tree()
        self.classes_output_mapping = self._calc_classes_output_mapping()
        model = NetworkInNetworkDecisioNetV2(cfg_name='CIFAR100_baseline_v2',
                                             classes_division=self.classes_division_tree,
                                             decisionet_cls=DecisioNetV2New)
        return model

    def _load_weights(self, model: nn.Module, weights_path: str):
        relevant_modules = [model.decisionet,
                            model.decisionet.left,
                            model.decisionet.left.left, model.decisionet.left.right,
                            model.decisionet.right,
                            model.decisionet.right.left, model.decisionet.right.right]
        depths = ['0', '1', '2', '2', '1', '2', '2']
        for m, w, d in zip(relevant_modules, weights_path, depths):
            state = torch.load(w, map_location=torch.device('cpu'))
            features_state_dict = {}
            binary_classifier_state_dict = {}
            for k, v in state['net'].items():
                if k.startswith('features.'):
                    if k.startswith(f'features.{d}'):
                        features_state_dict[k.replace(f'features.{d}.', '')] = v
                else:
                    if k.startswith('binary_classifier.'):
                        binary_classifier_state_dict[k.replace('binary_classifier.', '')] = v
            m.features.load_state_dict(features_state_dict)
            if not m.is_leaf:
                m.binary_selection_layer.load_state_dict(binary_classifier_state_dict)

    @staticmethod
    def _parse_weights_path(weights_path: str) -> Union[str, TextIO]:
        weights_paths = []
        for w in weights_path:
            weights_paths.append(NetworkInNetworkDecisioNetTrainerSigmaLossV3._parse_weights_path(w))
        return weights_paths

    def _feed_forward(self, inputs, targets):
        cls_targets, *sigma_targets = targets
        sigma_targets = torch.column_stack(sigma_targets)
        binarize = True  # False
        orig_outputs, sigmas = self.model(inputs, binarize=binarize)
        outputs = orig_outputs.clone()
        outputs[:, self.classes_output_mapping] = orig_outputs
        cls_loss = self.cls_criterion(outputs, cls_targets.long())
        # sigma_loss = self.sigma_criterion(sigmas, sigma_targets)
        sigma_loss = self.sigma_criterion(sigmas.double(), sigma_targets)
        combined_loss = cls_loss + self.beta * sigma_loss
        self.metrics_tracker.update(cls_loss, sigma_loss, combined_loss, outputs, cls_targets, sigmas, sigma_targets)
        return outputs, combined_loss


class NinDnPretrainedV2(NinDnPretrained):

    def _init_model(self):
        self.classes_division_tree = self._init_classes_division_tree()
        self.classes_output_mapping = self._calc_classes_output_mapping()
        model = NetworkInNetworkDecisioNetV3(cfg_name='CIFAR10_baseline_v2',  # cfg_name='CIFAR100_baseline_v3',
                                             classes_division=self.classes_division_tree,
                                             decisionet_cls=DecisioNetV2New,
                                             num_classes=self.num_classes)
        return model


class NetworkInNetworkDecisioNetTrainerSigmaLossRatios(NetworkInNetworkDecisioNetTrainer):

    def _init_classes_division_tree(self):
        labels_map = dict(LABELS_MAP[self.dataset_name])
        for k, v in labels_map.items():
            labels_map[k] = v[1:]
        tree = Node.init_from_dict(labels_map)
        return tree

    # noinspection PyTypeChecker
    def _calc_classes_output_mapping(self):
        num_leaves = len(self.classes_division_tree.leaves)
        indices = [None] * num_leaves
        level_order = self.classes_division_tree.level_order_encode()
        for node, code in level_order:
            if node.is_leaf():
                idx = int('0b' + ''.join(map(str, code)), 2)
                assert indices[idx] is None
                indices[idx] = torch.tensor(node.value)
        indices = torch.cat(indices)
        return indices

    def _init_model(self):
        self.classes_division_tree = self._init_classes_division_tree()
        self.classes_output_mapping = self._calc_classes_output_mapping()
        model = NetworkInNetworkDecisioNetV3(cfg_name=self.nin_cfg_name,  # cfg_name='CIFAR100_baseline_v3',
                                             classes_division=self.classes_division_tree,
                                             decisionet_cls=DecisioNetV2,
                                             num_classes=self.num_classes)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.normal_(0, 0.0)
        return model

    def _feed_forward(self, inputs, targets):
        cls_targets, *sigma_targets = targets
        sigma_targets = torch.column_stack(sigma_targets)
        binarize = self.always_binarize or random.random() > 0.5
        orig_outputs, sigmas = self.model(inputs, binarize=binarize)
        outputs = orig_outputs.clone()
        outputs[:, self.classes_output_mapping] = orig_outputs
        cls_loss = self.cls_criterion(outputs, cls_targets.long())
        sigma_loss = self.sigma_criterion(sigmas.double(), sigma_targets)
        combined_loss = cls_loss + self.beta * sigma_loss
        self.metrics_tracker.update(cls_loss, sigma_loss, combined_loss, outputs, cls_targets, sigmas, sigma_targets)
        return outputs, combined_loss


class NetworkInNetworkDecisioNetTrainerSigmaLossRatiosV2(NetworkInNetworkDecisioNetTrainer):

    def _init_classes_division_tree(self):
        labels_map = dict(LABELS_MAP[self.dataset_name])
        for k, v in labels_map.items():
            labels_map[k] = v[1:]
        tree = Node.init_from_dict(labels_map)
        return tree

    # noinspection PyTypeChecker
    def _calc_classes_output_mapping(self):
        num_leaves = len(self.classes_division_tree.leaves)
        indices = [None] * num_leaves
        level_order = self.classes_division_tree.level_order_encode()
        for node, code in level_order:
            if node.is_leaf():
                idx = int('0b' + ''.join(map(str, code)), 2)
                assert indices[idx] is None
                indices[idx] = torch.tensor(node.value)
        indices = torch.cat(indices)
        return indices

    def _init_model(self):
        self.classes_division_tree = self._init_classes_division_tree()
        model = NetworkInNetworkDecisioNetV4(cfg_name=self.nin_cfg_name,
                                             classes_division=self.classes_division_tree,
                                             num_classes=self.num_classes)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.normal_(0, 0.0)
        return model
