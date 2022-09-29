import os
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from data.datasets import DATA_ROOT_PATH, DATASETS
from data.transforms import BasicTransforms
from models.wide_resnet import wide_resnet28_10
from utils.binary_tree import Node
from utils.constants import NUM_CLASSES, INPUT_SIZE

np.random.seed(42)


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


class ClassesClusterer:
    def __init__(self,
                 dataset_name: str,
                 net: nn.Module,
                 ckpt_path: str,
                 confusion_matrix_output_path: str,
                 batch_size=64,
                 num_workers=-1,
                 extend_clusters=False,
                 extend_cluster_conf_threshold=0.1,
                 maximal_cluster_size_ratio=2 / 3,
                 normalize_confusion_matrix=False,
                 subsample_ratio: Optional[int] = None,
                 train: bool = False):
        """

        Args:
            dataset_name: Name of the current dataset
            net: net to use for clustering
            ckpt_path: a path to the checkpoint to load
            confusion_matrix_output_path: confusion matrix output path
            batch_size: batch size
            num_workers: number of workers for the dataloader
            extend_clusters: whether to extend the clusters, i.e., classes can be assigned to more than one cluster
            extend_cluster_conf_threshold: confidence threshold for extending the clusters
            maximal_cluster_size_ratio: maximal cluster size ratio allowed
            normalize_confusion_matrix: set to True to normalize the confusion matrix in the recursive clustering process
            subsample_ratio: subsample ratio for loading the part of the dataset
            train: whether to use train or test data
        """
        self.normalize_confusion_matrix = normalize_confusion_matrix
        self.maximal_cluster_size_ratio = maximal_cluster_size_ratio
        self.extend_cluster_conf_threshold = extend_cluster_conf_threshold
        self.extend_clusters = extend_clusters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net = net
        self.load_weights(ckpt_path)
        if num_workers < 0:
            num_workers = 0 if self.device == 'cpu' else 1
        self.data_loader = self.init_data_loader(dataset_name, batch_size, num_workers, train=train,
                                                 subsample_ratio=subsample_ratio)
        self.num_classes = NUM_CLASSES[dataset_name]
        self.confusion_matrix_output_path = confusion_matrix_output_path

    @staticmethod
    def balanced_subsample(dataset, num_samples_per_class, shuffle=True):
        assert num_samples_per_class <= np.min(np.bincount(dataset.targets)), "no double samples allowed"
        dataset.targets = np.array(dataset.targets)
        all_samples = []
        num_classes = len(np.unique(dataset.targets))
        for i in range(num_classes):
            cls_idx = np.where(dataset.targets == i)[0]
            cls_random_sample = np.random.choice(cls_idx, num_samples_per_class, replace=False)
            all_samples.extend(cls_random_sample.tolist())
        if shuffle:
            np.random.shuffle(all_samples)
        dataset.data = dataset.data[all_samples]
        dataset.targets = dataset.targets[all_samples].tolist()

    @staticmethod
    def init_data_loader(dataset_name, batch_size, num_workers, train=False, subsample_ratio: Optional[int] = None):
        basic_transforms = BasicTransforms(dataset_name, augment=False)

        dataset = DATASETS[dataset_name]
        data_set = dataset(root=DATA_ROOT_PATH, train=train, download=True, transform=basic_transforms.transform_test)
        if subsample_ratio is not None:
            assert 0 < subsample_ratio < 1
            num_sub_samples = int(np.min(np.bincount(data_set.targets)) * subsample_ratio)
            assert num_sub_samples > 0, "Got 0 subsample, something's wrong"
            print(f"Subsampling {num_sub_samples} samples of each class")
            ClassesClusterer.balanced_subsample(data_set, num_sub_samples)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers)
        return data_loader

    def load_weights(self, ckpt_path):
        map_location = torch.device('cpu') if self.device == 'cpu' else None
        state = torch.load(ckpt_path, map_location=map_location)
        self.net = self.net.to(self.device)
        state_dict = OrderedDict({k.replace('module.', ''): v for k, v in state['net'].items()})
        self.net.load_state_dict(state_dict)

    def hierarchical_binary_clustering(self,
                                       distance_matrix: torch.Tensor,
                                       tree_depth: int):

        clusters = []
        for d in range(tree_depth - 1, -1, -1):
            num_nodes_in_current_depth = 2 ** d
            clustering = AgglomerativeClustering(n_clusters=num_nodes_in_current_depth,
                                                 affinity='precomputed', linkage='average', compute_distances=True)
            clustering.fit(distance_matrix)
            labels = clustering.labels_
            new_clusters = []
            for i in range(num_nodes_in_current_depth):
                label_i_classes = np.where(labels == i)[0]
                new_clusters.append(Node(label_i_classes))
            while clusters:
                c = clusters.pop(0)
                for nc in new_clusters:
                    # noinspection PyUnresolvedReferences
                    if np.intersect1d(c.val, nc.val).size > 0:
                        if nc.left is None:
                            nc.left = c
                        elif nc.right is None:
                            nc.right = c
                        else:
                            print("Something went wrong!!!!")

            clusters = new_clusters
        assert len(clusters) == 1
        tree = clusters[0]
        return tree

    def recursive_binary_clustering(self,
                                    distance_matrix: torch.Tensor,
                                    confusion_matrix: torch.Tensor,
                                    num_repeats: int,
                                    tree: Node):
        if len(tree.value) == 1:
            tree.left = Node(tree.value)
            return
        clustering = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average',
                                             compute_distances=True)
        if self.normalize_confusion_matrix:
            confusion_matrix = confusion_matrix / confusion_matrix.sum(1, keepdims=True)
            distance_matrix = 1 - confusion_matrix
            distance_matrix.fill_diagonal_(0.0)
            distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
        clustering.fit(distance_matrix)
        labels = clustering.labels_
        this_level_classes = tree.value
        classes0_idx = np.where(labels == 0)[0]
        classes1_idx = np.where(labels == 1)[0]
        if self.extend_clusters:
            max_cluster_size = int(this_level_classes.size * self.maximal_cluster_size_ratio)
            classes0_idx, classes1_idx = self._extend_clusters([classes0_idx, classes1_idx], confusion_matrix,
                                                               max_cluster_size)
        classes0 = this_level_classes[classes0_idx]
        classes1 = this_level_classes[classes1_idx]
        tree.left = Node(classes0)
        tree.right = Node(classes1)
        if num_repeats == 1:
            return
        else:
            dm0 = distance_matrix[np.ix_(classes0_idx, classes0_idx)]
            dm1 = distance_matrix[np.ix_(classes1_idx, classes1_idx)]
            cm0 = confusion_matrix[np.ix_(classes0_idx, classes0_idx)]
            cm1 = confusion_matrix[np.ix_(classes1_idx, classes1_idx)]
            self.recursive_binary_clustering(dm0, cm0, num_repeats - 1, tree.left)
            self.recursive_binary_clustering(dm1, cm1, num_repeats - 1, tree.right)

    def run_clustering(self, print_clustering=True, recalculate_confusion_matrix=False, num_splits=2):
        F = self._calc_confusion_matrix(recalculate_confusion_matrix)
        D = 1 - F
        D.fill_diagonal_(0.0)
        D = 0.5 * (D + D.T)
        sigma_map, tree = self.cluster(D, F, verbose=print_clustering, num_splits=num_splits)
        return sigma_map

    def _calc_confusion_matrix(self, recalculate=False):
        if not recalculate and os.path.exists(self.confusion_matrix_output_path):
            F = torch.load(self.confusion_matrix_output_path)
        else:
            dirname = os.path.dirname(self.confusion_matrix_output_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            self.net.eval()
            confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
            with torch.no_grad():
                for inputs, targets in tqdm(self.data_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.net(inputs)
                    _, predicted = outputs.max(1)
                    for t, p in zip(targets.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
            print(confusion_matrix)
            F = confusion_matrix / confusion_matrix.sum(1, keepdims=True)  # normalized confusion matrix
            torch.save(F, self.confusion_matrix_output_path)
        return F

    # noinspection PyTypeChecker
    def _extend_clusters(self, cluster_members, confusion_matrix, max_cluster_size):
        threshold = self.extend_cluster_conf_threshold
        extended_cluster_members = []
        for curr_cluster in range(2):
            other_cluster = 1 - curr_cluster
            curr_cluster_members = cluster_members[curr_cluster]
            other_cluster_members = cluster_members[other_cluster]
            conf_mat_slice = confusion_matrix[np.ix_(other_cluster_members, curr_cluster_members)]
            curr_cluster_probs = torch.sum(conf_mat_slice, dim=1)
            candidates_indices = torch.where(curr_cluster_probs > threshold)[0]
            curr_cluster_new_candidates = other_cluster_members[candidates_indices]
            if isinstance(curr_cluster_new_candidates, np.integer):
                curr_cluster_new_candidates = np.array([curr_cluster_new_candidates])
            all_curr_cluster_members = np.concatenate([curr_cluster_members, curr_cluster_new_candidates])
            if max_cluster_size > curr_cluster_members.size:  # do not remove existing members!!
                all_curr_cluster_members = all_curr_cluster_members[:max_cluster_size]
            extended_cluster_members.append(all_curr_cluster_members)
        return extended_cluster_members

    def cluster(self, distance_matrix: torch.Tensor, confusion_matrix: torch.Tensor, verbose=True, num_splits=2):
        root_classes = np.arange(self.num_classes)
        tree = Node(root_classes)
        self.recursive_binary_clustering(distance_matrix, confusion_matrix, num_splits, tree)
        all_nodes = tree.level_order_encode()
        sigma_map = {}
        for node, code in all_nodes:
            if node.is_leaf():
                for v in node.val:
                    if v not in sigma_map:
                        sigma_map[v] = [code]
                    else:
                        sigma_map[v].append(code)

        sigma_map = dict(sorted(sigma_map.items()))
        for k, v in sigma_map.items():
            if isinstance(v, list) and len(v) == 2:
                a, b = v
                res = ()
                for i in range(len(a)):
                    val = a[i] if a[i] == b[i] else 0.5
                    res = res + (val,)
                sigma_map[k] = res
            else:
                sigma_map[k] = v[0]
        if verbose:
            print(sigma_map)
            print(tree)
        return sigma_map, tree


def download_checkpoint(file_name, run_path, dest_dir=None):
    api = wandb.apis.public.Api()
    api_run = api.run(run_path)
    if dest_dir is None:
        dest_dir = os.getcwd()
    path = os.path.join(dest_dir, file_name)
    if os.path.exists(path):
        return path

    files = api_run.files([file_name])
    if len(files) == 0:
        return None
    # if the file does not exist, the file has an md5 of 0
    if files[0].md5 == "0":
        raise ValueError("File {} not found in {}.".format(file_name, run_path))
    ###
    wandb.apis.util.download_file_from_url(path, files[0].url, api.api_key)
    return path


if __name__ == '__main__':
    dataset_name = 'CIFAR100'
    num_classes = NUM_CLASSES[dataset_name]
    num_in_channels = INPUT_SIZE[dataset_name][0]
    run_name, run_path = ('fluent-pine-8', "noamgot/CIFAR100_WideResNet_augment/39d3t5y3")
    ckpt_path = download_checkpoint(f'{run_name}_ckpt.pth', run_path=run_path)
    assert ckpt_path
    # net = NetworkInNetwork(num_classes=num_classes, num_in_channels=num_in_channels)
    net = wide_resnet28_10(num_classes=num_classes)
    confusion_matrix_output_path = f'/tmp/{run_name}.pt'
    clusterer = ClassesClusterer(dataset_name, net, ckpt_path, confusion_matrix_output_path,
                                 normalize_confusion_matrix=True, extend_clusters=False, train=False)
    # clusterer.run_clustering(recalculate_confusion_matrix=True)
    clusterer.run_clustering(num_splits=2, recalculate_confusion_matrix=True)
    print("done")
