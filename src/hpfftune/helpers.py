import inspect
import types
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, cast

import optuna
import pandas as pd
from sklearn.base import BaseEstimator


@dataclass
class MetaData:
    name: str

    @abstractmethod
    def param_map(self, trial: optuna.Trial) -> dict[str, Any]:
        pass

    @abstractmethod
    def regressor(self, params: dict[str, Any]) -> BaseEstimator:
        pass

    @abstractmethod
    def clean_data(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        pass


def get_calling_func_name() -> str:
    this = cast(types.FrameType, inspect.currentframe())
    parent = cast(types.FrameType, this.f_back)

    return parent.f_code.co_name
