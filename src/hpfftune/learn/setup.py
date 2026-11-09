import importlib
from collections.abc import Callable
from copy import deepcopy
from inspect import getmembers, isclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder

from hpfftune.helpers import MetaData, NumberParams
from hpfftune.learn import regressors


def wrapped_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_test_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, **train_test_params)


def get_classes(module) -> list[str]:
    return [
        cls.__name__
        for _, cls in getmembers(module, isclass)
        if cls.__module__ == module.__name__
        and cls.__base__ is not None
        and cls.__base__.__name__ == MetaData.__name__
    ]


def import_class(class_name: str, module_name: str) -> Callable:
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def get_regressor_metadata(module) -> dict[str, MetaData]:
    metadata_classes = get_classes(module)
    avail_regressors: dict[str, MetaData] = {}

    for name in metadata_classes:
        metadata_init: MetaData = import_class(name, module.__name__)()
        regressor_name = metadata_init.name
        avail_regressors[regressor_name] = metadata_init

    return avail_regressors


def select_params(
    trial: optuna.Trial,
    potential_params: dict[str, list[Any] | NumberParams],
) -> dict[str, Any]:
    selected_params: dict[str, Any] = {}
    for param_name, params in potential_params.items():
        if isinstance(params, list):
            try:
                suggested_param: Any = trial.suggest_categorical(param_name, params)
            except ValueError as e:
                print(f"{param_name}-{params}")
                raise e
        elif isinstance(params, dict):
            low = params["low"]
            high = params["high"]

            step = None
            if "step" in params:
                step = params["step"]

            ty = params["ty"]
            if ty is float:
                suggested_param = trial.suggest_float(param_name, low, high, step=step)
            elif ty is int:
                low = int(low)
                high = int(high)

                if step is None:
                    suggested_param = trial.suggest_int(param_name, low, high)
                else:
                    step = int(step)
                    suggested_param = trial.suggest_int(
                        param_name, low, high, step=step
                    )

        selected_params[param_name] = suggested_param

    return selected_params


class Objective:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def __call__(self, trial: optuna.Trial) -> float:
        avail_regressors = get_regressor_metadata(regressors)

        regressor_name = trial.suggest_categorical(
            "regressor", list(avail_regressors.keys())
        )

        regressor_metadata = avail_regressors[regressor_name]
        potential_params = regressor_metadata.param_map
        selected_params = select_params(trial, potential_params)
        regressor_init = regressor_metadata.regressor(selected_params)

        score: np.ndarray = cross_val_score(regressor_init, self.X, self.y)
        return score.mean()


def setup(dataset: pd.DataFrame, target: str):
    X: pd.DataFrame = dataset.drop([target])
    y: pd.Series = dataset[target]

    categorical_columns_subset = [
        "format",
        "task_type",
        "language",
        "no_caching",
        "parallel_backend",
        "access_kind",
        "unit_var",
        "unit_chunk",
    ]

    numerical_columns_subset = [
        "dimensions",
        "num_nodes",
        "num_ranks",
        "parallel",
        "total_filesize",
        "filesize_per_var",
        "filesize_per_chunk",
    ]

    X: pd.DataFrame = X[categorical_columns_subset + numerical_columns_subset]
    X[categorical_columns_subset] = X[categorical_columns_subset].astype("category")

    columns_to_encode: list[str] = X.select_dtypes(
        exclude=["float", "int"]
    ).columns.to_list()

    new_order = deepcopy(columns_to_encode)
    remainders: list[str] = X.select_dtypes(include=["float", "int"]).columns.to_list()
    new_order.extend(remainders)
    preprocessor = ColumnTransformer(
        transformers=[("oe", OrdinalEncoder(), columns_to_encode)],
        remainder="passthrough",
    )

    X: pd.DataFrame = pd.DataFrame(
        preprocessor.fit_transform(X),
        columns=new_order,
    )

    sampler = TPESampler(seed=10)
    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name="HPFFtune",
        load_if_exists=True,
    )
    study.optimize(Objective(X, y), n_trials=100, show_progress_bar=True)
