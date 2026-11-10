import sys
from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from hpfftune.helpers import MetaData


@dataclass
class HistGradBoostRegMetaData(MetaData):
    def __init__(self):
        self.name = "hist_grad_boost_reg"

    def param_map(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "quantile": trial.suggest_float("quantile", 0, 1),
            "learning_rate": trial.suggest_float("learning_rate", 0, 1),
            "max_iter": trial.suggest_int("max_iter", 100, 10_000, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 50),
            "max_depth": trial.suggest_int("max_depth", 1, sys.maxsize),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 0, 1),
            "max_features": trial.suggest_float("max_features", 0, 1),
            "max_bins": trial.suggest_int("max_bins", 1, 255),
            "warm_start": trial.suggest_categorical("warm_start", [False, True]),
            "early_stopping": trial.suggest_categorical(
                "early_stopping", ["auto", False, True]
            ),
            "scoring": trial.suggest_categorical(
                "scoring", ["loss", "neg_max_error", "r2", None]
            ),
            "validation_fraction": trial.suggest_float("validation_fraction", 0.1, 1),
            "n_iter_no_change": trial.suggest_int("n_iter_no_change", 10, 100, step=10),
            "tol": trial.suggest_float("tol", 1e-9, 1e-4),
        }

    def regressor(self, params: dict[str, Any]) -> BaseEstimator:
        return HistGradientBoostingRegressor(**params)

    def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


@dataclass
class RandomForestRegressorMetaData(MetaData):
    def __init__(self):
        self.name = "random_forest_reg"

    def param_map(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000, step=10),
            "criterion": trial.suggest_categorical(
                "criterion", ["squared_error", "absolute_error", "poisson"]
            ),
            "max_depth": trial.suggest_int("max_depth", 10, sys.maxsize),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_weight_fraction_leaf": trial.suggest_float(
                "min_weight_fraction_leaf", 0, 0.5
            ),
            "max_features": trial.suggest_float("max_features", 0, 1),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 50),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0, 1),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0, 1),
            "max_samples": trial.suggest_int("max_samples", 10, 100),
            "warm_start": trial.suggest_categorical("warm_start", [False, True]),
        }

    def regressor(self, params: dict[str, Any]) -> BaseEstimator:
        return RandomForestRegressor(**params)

    def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


@dataclass
class SVRMetaData(MetaData):
    def __init__(self):
        self.name = "support_vector_reg"

    def param_map(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            "degree": trial.suggest_int("degree", 1, 9),
            "gamma": trial.suggest_categorical("gamma", ["auto", "scale"]),
            "coef0": trial.suggest_float("coef0", 0, 1),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3),
            "C": trial.suggest_float("C", 0.1, 1),
            "epsilon": trial.suggest_float("epsilon", 0.1, 1),
            "shrinking": trial.suggest_categorical("shrinking", [False, True]),
        }

    def regressor(self, params: dict[str, Any]) -> BaseEstimator:
        return SVR(**params)

    def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.dropna(axis=1)


@dataclass
class SGDMetaData(MetaData):
    def __init__(self):
        self.name = "stochastic_gradient_descent_reg"

    def param_map(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "loss": trial.suggest_categorical(
                "loss",
                [
                    "squared_error",
                    "huber",
                    "epsilon_insensitive",
                    "squared_epsilon_insensitive",
                ],
            ),
            "penalty": trial.suggest_categorical(
                "penalty", ["l2", "l1", "elasticnet", None]
            ),
            "alpha": trial.suggest_float("alpha", 0.0001, sys.maxsize),
            "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 100, 10_000, step=100),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3),
            "shuffle": trial.suggest_categorical("shuffle", [True, False]),
            "epsilon": trial.suggest_float("epsilon", 0.1, sys.maxsize),
            "learning_rate": trial.suggest_categorical(
                "learning_rate",
                ["constant", "optimal", "invscaling", "adaptive", "pa1", "pa2"],
            ),
            "eta0": trial.suggest_float("eta0", 0.01, sys.maxsize),
            "early_stopping": trial.suggest_categorical(
                "early_stopping", [True, False]
            ),
            "validation_fraction": trial.suggest_float("validation_fraction", 0.1, 1),
            "n_iter_no_change": trial.suggest_int("n_iter_no_change", 10, 100),
            "warm_start": trial.suggest_categorical("warm_start", [True, False]),
            "average": trial.suggest_categorical("average", [True, False, 10]),
        }

    def regressor(self, params: dict[str, Any]) -> BaseEstimator:
        return SGDRegressor(**params)

    def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.dropna(axis=1)


@dataclass
class MLPMetaData(MetaData):
    def __init__(self):
        self.name = "multi_layer_perceptron_reg"

    def param_map(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "loss": trial.suggest_categorical("loss", ["squared_error", "poisson"]),
            "activation": trial.suggest_categorical(
                "activation", ["identity", "logistic", "tanh", "relu"]
            ),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
            "alpha": trial.suggest_float("alpha", 0.0001, sys.maxsize),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", ["constant", "invscaling", "adaptive"]
            ),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 0.001, 1),
            "power_t": trial.suggest_float("power_t", 0, 1),
            "max_iter": trial.suggest_int("max_iter", 100, 10_000, step=100),
            "shuffle": trial.suggest_categorical("shuffle", [True, False]),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3),
            "warm_start": trial.suggest_categorical("warm_start", [True, False]),
            "early_stopping": trial.suggest_categorical(
                "early_stopping", [True, False]
            ),
            "validation_fraction": trial.suggest_float("validation_fraction", 0.1, 1),
            "epsilon": trial.suggest_float("epsilon", 1e-8, 1e-3),
            "n_iter_no_change": trial.suggest_int("n_iter_no_change", 10, 100),
            "beta_1": trial.suggest_float("beta_1", 0.1, 0.9),
            "beta_2": trial.suggest_float("beta_2", 0.1, 0.999),
            "max_fun": trial.suggest_int("max_fun", 10_000, 20_000, step=1000),
        }

    def regressor(self, params: dict[str, Any]) -> BaseEstimator:
        return MLPRegressor(**params)

    def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.dropna(axis=1)
