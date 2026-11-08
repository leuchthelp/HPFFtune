import sys
from abc import ABC, abstractmethod
from typing import Any, NotRequired, TypedDict

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


class MetaData(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

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


class HistGradBoostRegMetaData(MetaData):
    @property
    def name(self) -> str:
        return "hist_grad_boost_reg"

    @property
    def param_map(
        self,
    ) -> dict[str, list[Any] | NumberParams]:
        return {
            "quantile": {"low": 0, "high": 1, "ty": float},
            "learning_rate": {"low": 0, "high": 1, "ty": float},
            "max_iter": {"low": 100, "high": 10_000, "step": 50, "ty": int},
            "max_leaf_nodes": {"low": 1, "high": sys.maxsize, "ty": int},
            "max_depth": {"low": 1, "high": sys.maxsize, "ty": int},
            "min_samples_leaf": {"low": 20, "high": 100, "ty": int},
            "l2_regularization": {"low": 0, "high": 1, "ty": float},
            "max_features": {"low": 0, "high": 1, "ty": float},
            "max_bins": {"low": 1, "high": 255, "ty": int},
            "categorical_features": [None, "from_dtype"],
            "monotonic_cst": [-1, 0, 1],
            "warm_start": [False, True],
            "early_stopping": ["auto", False, True],
            "scoring": ["loss", "neg_max_error", "r2"],
            "validation_fraction": {"low": 0.1, "high": 1, "ty": float},
            "n_iter_no_change": {"low": 10, "high": 100, "ty": int, "step": 10},
            "tol": [1e-7, 1e-6, 1e-5, 1e-4, 1e-8, 1e-9],
        }

    def regressor(self, params: dict[str, Any]) -> BaseEstimator:
        return HistGradientBoostingRegressor(**params)


class RandomForestRegressorMetaData(MetaData):
    @property
    def name(self) -> str:
        return "random_forest_reg"

    @property
    def param_map(
        self,
    ) -> dict[str, list[Any] | NumberParams]:
        return {}

    def regressor(self, params: dict[str, Any]) -> BaseEstimator:
        return RandomForestRegressor(**params)
