# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Vendored from pytorch-ts (Zalando Research) for incomplete local trees.

import time
from typing import Optional, Union

import torch
import torch.nn as nn
from gluonts.core.component import validated
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer:
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        maximum_learning_rate: float = 1e-2,
        clip_gradient: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        if device is None:
            self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        net.to(self.device)
        optimizer = Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )

        for epoch_no in range(self.epochs):
            tic = time.time()
            cumm_epoch_loss = 0.0
            total = self.num_batches_per_epoch - 1

            with tqdm(train_iter, total=total) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()
                    inputs = [v.to(self.device) for v in data_entry.values()]
                    output = net(*inputs)
                    loss = output[0] if isinstance(output, (list, tuple)) else output
                    cumm_epoch_loss += loss.item()
                    avg_epoch_loss = cumm_epoch_loss / batch_no
                    it.set_postfix(
                        {
                            "epoch": f"{epoch_no + 1}/{self.epochs}",
                            "avg_loss": avg_epoch_loss,
                        },
                        refresh=False,
                    )
                    loss.backward()
                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)
                    optimizer.step()
                    lr_scheduler.step()
                    if self.num_batches_per_epoch == batch_no:
                        break

            if validation_iter is not None:
                cumm_epoch_loss_val = 0.0
                with tqdm(validation_iter, total=total, colour="green") as it:
                    for batch_no, data_entry in enumerate(it, start=1):
                        inputs = [v.to(self.device) for v in data_entry.values()]
                        with torch.no_grad():
                            output = net(*inputs)
                            loss = (
                                output[0]
                                if isinstance(output, (list, tuple))
                                else output
                            )
                        cumm_epoch_loss_val += loss.item()
                        avg_epoch_loss_val = cumm_epoch_loss_val / batch_no
                        it.set_postfix(
                            {
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                "avg_loss": avg_epoch_loss,
                                "avg_val_loss": avg_epoch_loss_val,
                            },
                            refresh=False,
                        )
                        if self.num_batches_per_epoch == batch_no:
                            break

            _ = time.time() - tic
