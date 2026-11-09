import inspect
import types
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict, cast

from sklearn.base import BaseEstimator


@dataclass
class MetaData:
    name: str

    @property
    @abstractmethod
    def param_map(
        self,
    ) -> dict[str, list[Any] | NumberParams]:
        pass

    @abstractmethod
    def regressor(self, params: dict[str, Any]) -> BaseEstimator:
        pass


class NumberParams(TypedDict):
    low: float | int
    high: float | int
    step: NotRequired[float | int]
    ty: type


def get_calling_func_name() -> str:
    this = cast(types.FrameType, inspect.currentframe())
    parent = cast(types.FrameType, this.f_back)

    return parent.f_code.co_name
