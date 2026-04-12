# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Vendored from pytorch-ts (Zalando Research).

from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.multiprocessing as torch_mp
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.env import env
from gluonts.itertools import maybe_len
from gluonts.model.estimator import Estimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import SelectFields, Transformation

from pts import Trainer
from pts.model.utils import get_module_forward_input_names
from pts.dataset.loader import TransformedIterableDataset
from torch.utils.data import DataLoader


class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: nn.Module
    predictor: PyTorchPredictor


class PyTorchEstimator(Estimator):
    @validated()
    def __init__(
        self, trainer: Trainer, lead_time: int = 0, dtype: np.dtype = np.float32
    ) -> None:
        super().__init__(lead_time=lead_time)
        self.trainer = trainer
        self.dtype = dtype

    def create_transformation(self) -> Transformation:
        raise NotImplementedError

    def create_instance_splitter(self, mode: str) -> Transformation:
        raise NotImplementedError

    def create_training_network(self, device: torch.device) -> nn.Module:
        raise NotImplementedError

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: nn.Module,
        device: torch.device,
    ) -> PyTorchPredictor:
        raise NotImplementedError

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        # Avoid oversubscribing CPU cores across workers (esp. with spawn + OpenMP/MKL).
        torch.set_num_threads(1)

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        **kwargs,
    ) -> TrainOutput:
        transformation = self.create_transformation()
        trained_net = self.create_training_network(self.trainer.device)
        input_names = get_module_forward_input_names(trained_net)

        with env._let(max_idle_transforms=maybe_len(training_data) or 0):
            training_instance_splitter = self.create_instance_splitter("training")
            training_iter_dataset = TransformedIterableDataset(
                dataset=training_data,
                transform=transformation
                + training_instance_splitter
                + SelectFields(input_names),
                is_train=True,
                shuffle_buffer_length=shuffle_buffer_length,
                cache_data=cache_data,
            )

            data_loader_kwargs = dict(
                batch_size=self.trainer.batch_size,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=self._worker_init_fn if num_workers > 0 else None,
                **kwargs,
            )
            if num_workers > 0:
                data_loader_kwargs["prefetch_factor"] = prefetch_factor
                # Avoid tearing down workers each epoch (IterableDataset + multi-epoch loop
                # otherwise shows long 0% stalls and GPU idle at every epoch boundary).
                data_loader_kwargs.setdefault("persistent_workers", True)
                # Default fork() after the parent has touched CUDA is unsafe; workers then
                # intermittently abort (SIGABRT / "terminate called without an active exception").
                data_loader_kwargs.setdefault(
                    "multiprocessing_context", torch_mp.get_context("spawn")
                )
            training_data_loader = DataLoader(
                training_iter_dataset,
                **data_loader_kwargs,
            )

        validation_data_loader = None
        if validation_data is not None:
            with env._let(max_idle_transforms=maybe_len(validation_data) or 0):
                validation_instance_splitter = self.create_instance_splitter(
                    "validation"
                )
                validation_iter_dataset = TransformedIterableDataset(
                    dataset=validation_data,
                    transform=transformation
                    + validation_instance_splitter
                    + SelectFields(input_names),
                    is_train=True,
                    cache_data=cache_data,
                )
                val_kwargs = dict(
                    batch_size=self.trainer.batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    worker_init_fn=self._worker_init_fn if num_workers > 0 else None,
                    **kwargs,
                )
                if num_workers > 0:
                    val_kwargs["prefetch_factor"] = prefetch_factor
                    val_kwargs.setdefault("persistent_workers", True)
                    val_kwargs.setdefault(
                        "multiprocessing_context", torch_mp.get_context("spawn")
                    )
                validation_data_loader = DataLoader(validation_iter_dataset, **val_kwargs)

        self.trainer(
            net=trained_net,
            train_iter=training_data_loader,
            validation_iter=validation_data_loader,
        )
        return TrainOutput(
            transformation=transformation,
            trained_net=trained_net,
            predictor=self.create_predictor(
                transformation, trained_net, self.trainer.device
            ),
        )

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        **kwargs,
    ) -> PyTorchPredictor:
        return self.train_model(
            training_data,
            validation_data,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            **kwargs,
        ).predictor
