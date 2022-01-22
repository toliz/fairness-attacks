from torch import Tensor
from ..datamodules import Dataset


def project(point: Tensor, beta: dict) -> Tensor:
    raise NotImplementedError()


def defense_fn(dataset: Dataset, beta: dict) -> Dataset:
    raise NotImplementedError()


def get_defense_params(dataset: Dataset) -> dict:
    raise NotImplementedError()
