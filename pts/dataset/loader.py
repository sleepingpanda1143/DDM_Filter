# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Vendored from pytorch-ts (Zalando Research).

from __future__ import annotations

from typing import Optional

from gluonts.dataset.common import Dataset
from gluonts.itertools import Cached, Cyclic, PseudoShuffled
from gluonts.transform import Transformation, TransformedDataset
from torch.utils.data import IterableDataset


class TransformedIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        is_train: bool = True,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
    ):
        super().__init__()
        self.shuffle_buffer_length = shuffle_buffer_length
        base = Cyclic(dataset) if not cache_data else Cached(Cyclic(dataset))
        self.transformed_dataset = TransformedDataset(
            base,
            transformation=transform,
            is_train=is_train,
        )

    def __iter__(self):
        if self.shuffle_buffer_length is None:
            return iter(self.transformed_dataset)
        shuffled = PseudoShuffled(
            self.transformed_dataset,
            shuffle_buffer_length=self.shuffle_buffer_length,
        )
        return iter(shuffled)
