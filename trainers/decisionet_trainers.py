import random

import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt

# from custom_layers.losses import WeightedMSELoss
from data.datasets import FilteredRelabeledDatasets
from models.decisionet import NetworkInNetworkDecisioNet, NIN_CFG, WideResNetDecisioNet, WRESNET_STAGE_SIZES
from trainers.basic_trainer import BasicTrainer
from utils.constants import LABELS_MAP, CLASSES_NAMES, INPUT_SIZE, NUM_CLASSES
from utils.metrics_tracker import SigmaLossMetricsTracker


class DecisioNetTrainer(BasicTrainer):

    def __init__(self):
        super().__init__()
        # sigma_weights = self._init_sigma_weights()
        self.sigma_criterion = nn.MSELoss()  # WeightedMSELoss(sigma_weights)
        self.metrics_tracker = SigmaLossMetricsTracker(self.include_top5)

    def _init_model(self):
        raise NotImplementedError

    def _init_config_attributes(self):
        super()._init_config_attributes()
        self.beta = self.config['beta']
        self.always_binarize = self.config['always_binarize']

    def init_data_sets(self):
        labels_map = dict(LABELS_MAP[self.dataset_name])
        return FilteredRelabeledDatasets(self.transforms, use_validation=self.use_validation,
                                         classes_indices=self.classes_indices,
                                         labels_map=labels_map,
                                         dataset_name=self.dataset_name)

    def _feed_forward(self, inputs, targets):
        cls_targets, *sigma_targets = targets
        sigma_targets = torch.column_stack(sigma_targets)
        binarize = self.always_binarize or random.random() > 0.5
        outputs, sigmas = self.model(inputs, binarize=binarize)
        cls_loss = self.cls_criterion(outputs, cls_targets.long())
        sigma_loss = self.sigma_criterion(sigmas, sigma_targets.float())
        combined_loss = cls_loss + self.beta * sigma_loss
        self.metrics_tracker.update(cls_loss, sigma_loss, combined_loss, outputs, cls_targets, sigmas, sigma_targets)
        return outputs, combined_loss

    def _single_epoch(self, epoch: int, train_test_val: str):
        norm_acc, norm_loss = super()._single_epoch(epoch, train_test_val)
        if self.use_wandb:
            log_dict = {f"{train_test_val}_cls_loss": self.metrics_tracker.get_norm_cls_loss(),
                        f"{train_test_val}_sigma_loss": self.metrics_tracker.get_norm_sigma_loss(),
                        f"{train_test_val}_sigma_accuracy": 100. * self.metrics_tracker.get_norm_sigma_acc()}
            wandb.log(log_dict, step=epoch + 1)
        return norm_acc, norm_loss

    def input_and_targets_to_device(self, inputs, targets):
        inputs = inputs.to(self.device)
        for i in range(len(targets)):
            targets[i] = targets[i].to(self.device)
        return inputs, targets

    def init_parser(self):
        parser = super().init_parser()
        parser.add_argument('--beta', type=float, help='weight for the sigma loss', default=0.0)
        parser.add_argument('--always_binarize', action='store_true',
                            help='do not use non-binary values in the binarization layer (i.e., perform only hard routing)')
        return parser

    def register_hooks(self, activation_dict):
        def get_activation(activations_dict):
            def hook(model, input, output):
                predictions = output[0].detach()
                predictions = predictions.argmax(dim=1)
                sigma = output[1].detach()
                activations_dict['predictions'] = torch.cat([activations_dict['predictions'], predictions])
                activations_dict['sigma'] = torch.cat([activations_dict['sigma'], sigma])

            return hook

        hook_handles = []
        for name, layer in self.model.named_modules():
            if name == '':
                activation_dict['predictions'] = torch.Tensor([])
                activation_dict['sigma'] = torch.Tensor([])
                handle = layer.register_forward_hook(get_activation(activation_dict))
                hook_handles.append(handle)
        return hook_handles

    # noinspection PyTypeChecker
    def evaluate(self):
        activations_dict = {}
        self.register_hooks(activations_dict)
        super().evaluate()
        # norm_acc, _ = self._test_single_epoch(0)
        targets = torch.tensor(self.datasets.test_set.targets)
        predictions = activations_dict['predictions']
        sigmas = activations_dict['sigma']
        cls_targets = targets[:, 0]
        sigma_targets = targets[:, 1:]

        cls_acc = torch.sum(predictions == cls_targets) / targets.size(0)
        print(f"Class accuracy: {cls_acc * 100.}")
        sigma_diffs = (sigmas == sigma_targets)
        encoding = {0: 'both wrong', 1: 'first correct', 2: 'second correct', 3: 'both correct'}
        encoded_results = torch.sum(sigma_diffs * torch.tensor([1., 2.]), dim=1)
        for code, s in encoding.items():
            print(f'{s}: {torch.sum(encoded_results == code).item()}')
        num_images = 0
        for cls in self.classes_indices:
            if num_images % 10 == 0:
                plt.figure()
            plt.subplot(5, 2, num_images % 10 + 1)

            print("***********************************")
            class_name = CLASSES_NAMES[self.dataset_name][cls]
            plt.title(class_name)
            print(f"Class: {class_name}")
            cls_idx = torch.where(cls_targets == cls)[0]
            cls_acc = torch.sum(predictions[cls_idx] == cls) / cls_idx.size(0)
            print(f"Accuracy: {cls_acc * 100.}")
            sigma_diffs = (sigmas[cls_idx] == sigma_targets[cls_idx])
            encoded_results = torch.sum(sigma_diffs * torch.tensor([1., 2.]), dim=1)
            results = []
            for code, s in encoding.items():
                correct = torch.sum(encoded_results == code).item()
                results.append(correct)
                print(f'{s}: {correct}')
            plt.bar(list(encoding.values()), results)
            num_images += 1
        plt.show()
        plt.tight_layout()

    # noinspection PyTypeChecker
    def sigma_analysis(self):
        activations_dict = {}
        self.register_hooks(activations_dict)
        super().evaluate()
        targets = torch.tensor(self.datasets.test_set.targets)
        sigmas = activations_dict['sigma'].round()
        cls_targets = targets[:, 0]
        sigma_targets = targets[:, 1:]

        removed = []
        new_labels_map = {k: v[1:] for k, v in LABELS_MAP[self.dataset_name].items()}
        for s in range(2):
            soft_clustering_cands = []
            for cls in self.classes_indices:
                if cls in removed:
                    continue
                class_name = CLASSES_NAMES[self.dataset_name][cls]
                cls_idx = torch.where(cls_targets == cls)[0]
                results = []
                for i in range(2):
                    correct_until_split_idx = \
                        torch.where(torch.all(sigma_targets[cls_idx, :s] == sigmas[cls_idx, :s], dim=1))[0]
                    num_i = torch.sum(sigmas[cls_idx[correct_until_split_idx], s] == i).item()
                    results.append(num_i)
                if min(results) > 0.2 * sum(results):
                    soft_clustering_cands.append((class_name, cls, results))
                    plt.figure()
                    plt.bar(['0', '1'], results)
                    plt.title(f'{class_name} - split {s}')
            plt.show()
            print(f"SOFT CLUSTERING CANDS - SPLIT {s}:")
            for cls_name, cls, res in soft_clustering_cands:
                print(cls_name, f'({cls})', '- results:', res)
                orig_labels = new_labels_map[cls]
                new_labels = orig_labels[:s] + (0.5,) * (len(orig_labels) - s)
                new_labels_map[cls] = new_labels
                removed.append(cls)
        print(new_labels_map)

    def _init_sigma_weights(self):
        num_sigma_labels_per_samples = len(self.datasets.train_set.targets[0]) - 1
        sigma_weights = torch.tensor([0.5 ** i for i in range(num_sigma_labels_per_samples)]).to(self.device)
        return sigma_weights


class NetworkInNetworkDecisioNetTrainer(DecisioNetTrainer):

    def _init_model(self):
        num_in_channels = INPUT_SIZE[self.dataset_name][0]
        model = NetworkInNetworkDecisioNet(cfg_name=self.nin_cfg_name, num_in_channels=num_in_channels)
        return model

    def init_parser(self):
        parser = super().init_parser()
        parser.add_argument('--nin_cfg_name', type=str, default='10_baseline', help='Name of the NiN config')
        return parser

    def _init_config_attributes(self):
        super()._init_config_attributes()
        self.nin_cfg_name = self.config['nin_cfg_name']

    def init_data_sets(self):
        labels_map = dict(LABELS_MAP[self.dataset_name])
        num_blocks = len(NIN_CFG[self.nin_cfg_name])
        for k, v in labels_map.items():
            labels_map[k] = v[:num_blocks]
        return FilteredRelabeledDatasets(self.transforms, use_validation=self.use_validation,
                                         classes_indices=self.classes_indices,
                                         labels_map=labels_map,
                                         dataset_name=self.dataset_name)


class WideResNetDecisioNetTrainer(DecisioNetTrainer):

    def init_transforms(self, padding_mode='constant'):
        return super().init_transforms(padding_mode='reflect')

    def init_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [60, 120, 160], gamma=0.2, verbose=True)

    def lr_scheduler_step(self, epoch=-1, train_acc=None, train_loss=None, test_acc=None, test_loss=None):
        self.lr_scheduler.step()

    def init_optimizer(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                    momentum=0.9, weight_decay=5e-4, nesterov=True)
        return optimizer

    def _init_model(self):
        num_in_channels = INPUT_SIZE[self.dataset_name][0]
        num_classes = NUM_CLASSES[self.dataset_name]
        model = WideResNetDecisioNet(num_in_channels=num_in_channels, num_classes=num_classes, cfg_name=self.wrn_cfg_name)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return model

    def init_parser(self):
        parser = super().init_parser()
        parser.add_argument('--wrn_cfg_name', type=str, default='100_baseline', help='Name of the Wide-ResNet config')
        return parser

    def _init_config_attributes(self):
        super()._init_config_attributes()
        self.wrn_cfg_name = self.config['wrn_cfg_name']

    def init_data_sets(self):
        labels_map = dict(LABELS_MAP[f'{self.dataset_name}_WRN'])
        num_levels_in_tree = len(WRESNET_STAGE_SIZES[self.wrn_cfg_name])
        for k, v in labels_map.items():
            labels_map[k] = v[:num_levels_in_tree]
        return FilteredRelabeledDatasets(self.transforms, use_validation=self.use_validation,
                                         classes_indices=self.classes_indices,
                                         labels_map=labels_map,
                                         dataset_name=self.dataset_name)


if __name__ == '__main__':
    # trainer = NetworkInNetworkDecisioNetTrainer()
    trainer = WideResNetDecisioNetTrainer()
    trainer.train_model()
    # trainer.evaluate()
