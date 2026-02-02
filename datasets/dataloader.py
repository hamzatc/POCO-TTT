import os
import os.path as osp
import torch
import logging
import random
import numpy as np

from torch.utils.data import DataLoader, Subset

from configs.configs import BaseConfig, NeuralPredictionConfig, DatasetConfig
import datasets
from configs.config_global import ROOT_DIR, DEVICE

class DatasetIters(object):

    def __init__(self, config: NeuralPredictionConfig, phase: str):
        """
        Initialize a list of data loaders, compatible with multi-dataset/multi-task training.
        if only one dataset is specified, return a list containing one data loader
        """
        if type(config.dataset) is list or type(config.dataset) is tuple:
            dataset_list = config.dataset
        elif type(config.dataset) is str:
            dataset_list = [config.dataset]
        else:
            raise NotImplementedError('Dataset config not recognized')

        self.num_datasets = len(dataset_list)
        self.phase = phase
        assert len(config.mod_w) == self.num_datasets, 'mod_w and dataset len must match'

        self.data_iters = []
        self.iter_lens = []
        self.min_iter_len = None

        self.data_loaders = []
        self.input_sizes = []
        self.unit_types = []

        for dataset, label in zip(dataset_list, config.dataset_label):
            data_loader, input_size, unit_type = init_single_dataset(dataset, phase, config.dataset_config[label])
            self.data_loaders.append(data_loader)
            self.input_sizes.append(input_size)
            self.unit_types.append(unit_type)

        self.reset()

    def reset(self):
        # recreate iterator for each of the dataset
        self.data_iters = []
        self.iter_lens = []
        for data_l in self.data_loaders:
            data_iter = iter(data_l)
            self.data_iters.append(data_iter)
            self.iter_lens.append(len(data_iter))
        self.min_iter_len = min(self.iter_lens)

    def get_baselines(self, key='session_copy_mse'):
        assert self.phase != 'train', 'Baselines are only computed for test and val'
        all_baselines = []
        for loader in self.data_loaders:
            all_baselines.extend(loader.dataset.baseline[key])
        return all_baselines

def init_single_dataset(dataset_name: str, phase: str, config: DatasetConfig):
    collate_f = None
    train_flag = phase == 'train'
    input_size = None

    if dataset_name == 'zebrafish':
        dataset = datasets.Zebrafish(config, phase=phase)
    elif dataset_name == 'zebrafishahrens':
        dataset = datasets.ZebrafishAhrens(config, phase=phase)
    elif dataset_name == 'simulation':
        dataset = datasets.Simulation(config, phase=phase)
    elif dataset_name == 'celegans':
        dataset = datasets.Celegans(config, phase=phase)
    elif dataset_name == 'celegansflavell':
        dataset = datasets.CelegansFlavell(config, phase=phase)
    elif dataset_name == 'mice':
        dataset = datasets.Mice(config, phase=phase)
    else:
        raise NotImplementedError('Dataset not implemented')

    collate_f = dataset.collate_fn
    input_size = dataset.input_size
    unit_types = dataset.unit_types
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=train_flag or config.shuffle_test,
                             num_workers=config.num_workers, collate_fn=collate_f)

    return data_loader, input_size, unit_types


class SessionDatasetIters:
    """
    Dataset iterator that provides session-level access for meta-learning.
    Supports sampling sessions, splitting into support/query sets, and creating meta-batches.
    """

    def __init__(self, config: NeuralPredictionConfig, phase: str):
        """
        Initialize session-level data access.

        Args:
            config: Configuration object with dataset settings
            phase: 'train', 'val', or 'test'
        """
        if type(config.dataset) is list or type(config.dataset) is tuple:
            dataset_list = config.dataset
        elif type(config.dataset) is str:
            dataset_list = [config.dataset]
        else:
            raise NotImplementedError('Dataset config not recognized')

        self.num_datasets = len(dataset_list)
        self.phase = phase
        self.config = config

        # Store datasets and their metadata
        self.datasets = []
        self.input_sizes = []
        self.unit_types = []
        self.session_to_dataset = []  # Maps global session idx to (dataset_idx, local_session_idx)

        global_session_idx = 0
        for dataset_idx, (dataset_name, label) in enumerate(zip(dataset_list, config.dataset_label)):
            dataset = self._init_dataset(dataset_name, phase, config.dataset_config[label])
            self.datasets.append(dataset)
            self.input_sizes.append(dataset.input_size)
            self.unit_types.append(dataset.unit_types)

            # Map sessions
            for local_idx in range(dataset.num_sessions):
                self.session_to_dataset.append((dataset_idx, local_idx))
                global_session_idx += 1

        self.num_sessions = global_session_idx
        self.support_ratio = getattr(config, 'support_ratio', 0.7)

        # Build session indices: which data indices belong to each session
        self._build_session_indices()

    def _init_dataset(self, dataset_name: str, phase: str, config: DatasetConfig):
        """Initialize a single dataset"""
        if dataset_name == 'zebrafish':
            dataset = datasets.Zebrafish(config, phase=phase)
        elif dataset_name == 'zebrafishahrens':
            dataset = datasets.ZebrafishAhrens(config, phase=phase)
        elif dataset_name == 'simulation':
            dataset = datasets.Simulation(config, phase=phase)
        elif dataset_name == 'celegans':
            dataset = datasets.Celegans(config, phase=phase)
        elif dataset_name == 'celegansflavell':
            dataset = datasets.CelegansFlavell(config, phase=phase)
        elif dataset_name == 'mice':
            dataset = datasets.Mice(config, phase=phase)
        else:
            raise NotImplementedError(f'Dataset {dataset_name} not implemented')
        return dataset

    def _build_session_indices(self):
        """Build a mapping from session IDs to data indices"""
        self.session_indices = {}  # session_id -> list of (dataset_idx, data_idx)

        for dataset_idx, dataset in enumerate(self.datasets):
            # Track which session each data point belongs to
            for data_idx in range(len(dataset)):
                _, info = dataset[data_idx]
                local_session_idx = info['session_idx']

                # Find global session idx
                global_session_idx = None
                for gsi, (di, lsi) in enumerate(self.session_to_dataset):
                    if di == dataset_idx and lsi == local_session_idx:
                        global_session_idx = gsi
                        break

                if global_session_idx is not None:
                    if global_session_idx not in self.session_indices:
                        self.session_indices[global_session_idx] = []
                    self.session_indices[global_session_idx].append((dataset_idx, data_idx))

    def get_session(self, session_id: int):
        """
        Get all data for a specific session.

        Args:
            session_id: Global session index

        Returns:
            List of (data, info) tuples for this session
        """
        if session_id not in self.session_indices:
            return []

        dataset_idx, _ = self.session_to_dataset[session_id]
        dataset = self.datasets[dataset_idx]

        session_data = []
        for _, data_idx in self.session_indices[session_id]:
            data, info = dataset[data_idx]
            session_data.append((data, info))

        return session_data

    def get_support_query_split(self, session_id: int, support_ratio: float = None):
        """
        Split a session's data into support and query sets.

        Args:
            session_id: Global session index
            support_ratio: Fraction for support set (default: self.support_ratio)

        Returns:
            (support_data, query_data) - each is a list of (data, info) tuples
        """
        if support_ratio is None:
            support_ratio = self.support_ratio

        session_data = self.get_session(session_id)
        if len(session_data) == 0:
            return [], []

        # Shuffle and split
        indices = list(range(len(session_data)))
        random.shuffle(indices)

        split_idx = max(1, int(len(indices) * support_ratio))
        support_indices = indices[:split_idx]
        query_indices = indices[split_idx:]

        # Ensure at least one sample in each set
        if len(query_indices) == 0 and len(support_indices) > 1:
            query_indices = [support_indices.pop()]

        support_data = [session_data[i] for i in support_indices]
        query_data = [session_data[i] for i in query_indices]

        return support_data, query_data

    def sample_meta_batch(self, batch_size: int = None):
        """
        Sample a batch of sessions for meta-training.

        Args:
            batch_size: Number of sessions to sample (default: config.meta_batch_size)

        Returns:
            List of session IDs
        """
        if batch_size is None:
            batch_size = getattr(self.config, 'meta_batch_size', 4)

        available_sessions = [
            sid for sid in range(self.num_sessions)
            if sid in self.session_indices and len(self.session_indices[sid]) > 1
        ]

        if len(available_sessions) == 0:
            raise ValueError("No sessions with sufficient data for meta-learning")

        batch_size = min(batch_size, len(available_sessions))
        return random.sample(available_sessions, batch_size)

    def create_batch_from_data(self, data_list, session_id: int):
        """
        Create a batched input from a list of data samples for a specific session.

        Args:
            data_list: List of (data, info) tuples
            session_id: Global session index

        Returns:
            (input_list, target_list, info_list) formatted for model input
        """
        if len(data_list) == 0:
            return None

        dataset_idx, local_session_idx = self.session_to_dataset[session_id]
        dataset = self.datasets[dataset_idx]

        # Use the dataset's collate function logic
        input_size = dataset.input_size[local_session_idx]
        pred_length = dataset.pred_length
        seq_length = dataset.seq_length

        # Stack data into tensors
        batch_data = torch.stack([
            torch.from_numpy(data).float() for data, _ in data_list
        ], dim=1)  # L, B, D

        input_tensor = batch_data[:-pred_length] if pred_length > 0 else batch_data
        target_tensor = batch_data[1:]

        # Create input/target lists with placeholders for other sessions
        num_sessions_in_dataset = len(dataset.input_size)
        input_list = []
        target_list = []
        info_list = []

        for sess_idx in range(num_sessions_in_dataset):
            if sess_idx == local_session_idx:
                input_list.append(input_tensor.to(DEVICE))
                target_list.append(target_tensor.to(DEVICE))
                info_list.append({
                    'session_idx': sess_idx,
                    'normalize_coef': dataset.normalize_coef[sess_idx]
                })
            else:
                # Empty placeholder for other sessions
                other_size = dataset.input_size[sess_idx]
                input_list.append(torch.zeros(seq_length - pred_length, 0, other_size, device=DEVICE))
                target_list.append(torch.zeros(seq_length - 1, 0, other_size, device=DEVICE))
                info_list.append({'session_idx': sess_idx})

        return input_list, target_list, info_list

    def get_session_input_size(self, session_id: int):
        """Get the input size (number of neurons) for a session"""
        dataset_idx, local_session_idx = self.session_to_dataset[session_id]
        return self.datasets[dataset_idx].input_size[local_session_idx]

    def get_baselines(self, key='session_copy_mse'):
        """Get baseline performance metrics for all sessions"""
        assert self.phase != 'train', 'Baselines are only computed for test and val'
        all_baselines = []
        for dataset in self.datasets:
            all_baselines.extend(dataset.baseline[key])
        return all_baselines
