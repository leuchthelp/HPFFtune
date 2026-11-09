import sys
from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from hpfftune.helpers import MetaData, NumberParams


@dataclass
class HistGradBoostRegMetaData(MetaData):
    def __init__(self):
        self.name = "hist_grad_boost_reg"

    @property
    def param_map(
        self,
    ) -> dict[str, list[Any] | NumberParams]:
        return {
            "quantile": {"low": 0, "high": 1, "ty": float},
            "learning_rate": {"low": 0, "high": 1, "ty": float},
            "max_iter": {"low": 100, "high": 10_000, "step": 50, "ty": int},
            "max_leaf_nodes": {"low": 2, "high": 50, "ty": int},
            "max_depth": {"low": 1, "high": sys.maxsize, "ty": int},
            "min_samples_leaf": {"low": 20, "high": 100, "ty": int},
            "l2_regularization": {"low": 0, "high": 1, "ty": float},
            "max_features": {"low": 0, "high": 1, "ty": float},
            "max_bins": {"low": 1, "high": 255, "ty": int},
            "warm_start": [False, True],
            "early_stopping": [False, True],
            "scoring": ["loss", "neg_max_error", "r2"],
            "validation_fraction": {"low": 0.1, "high": 1, "ty": float},
            "n_iter_no_change": {"low": 10, "high": 100, "ty": int, "step": 10},
            "tol": {"low": 1e-9, "high": 1e-4, "ty": float},
        }

    def regressor(self, params: dict[str, Any]) -> BaseEstimator:
        return HistGradientBoostingRegressor(**params)


@dataclass
class RandomForestRegressorMetaData(MetaData):
    def __init__(self):
        self.name = "random_forest_reg"

    @property
    def param_map(
        self,
    ) -> dict[str, list[Any] | NumberParams]:
        return {
            "n_estimators": {"low": 10, "high": 1000, "step": 10, "ty": int},
            "criterion": ["squared_error", "absolute_error", "poisson"],
            "max_depth": {"low": 1, "high": sys.maxsize, "ty": int},
            "min_samples_split": {"low": 2, "high": 20, "ty": int},
            "min_samples_leaf": {"low": 1, "high": 20, "ty": int},
            "min_weight_fraction_leaf": {"low": 0, "high": 0.5, "ty": float},
            "max_features": {"low": 0, "high": 1, "ty": float},
            "max_leaf_nodes": {"low": 2, "high": 50, "ty": int},
            "min_impurity_decrease": {"low": 0, "high": 1, "ty": float},
            "ccp_alpha": {"low": 0, "high": 1, "ty": float},
            "max_samples": {"low": 10, "high": 100, "ty": int},
            "warm_start": [False, True],
        }

    def regressor(self, params: dict[str, Any]) -> BaseEstimator:
        return RandomForestRegressor(**params)
