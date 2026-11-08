from inspect import getmembers, getmodule, isclass
from typing import Any, TypeGuard

import numpy as np
import optuna
import pandas as pd
from narwhals import Object
from sklearn.model_selection import cross_val_score, train_test_split

from hpfftune.learn import regressors
from hpfftune.learn.regressors import MetaData, NumberParams


def wrapped_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_test_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, **train_test_params)


def ismetadata(object: Object) -> TypeGuard[MetaData]:
    return isclass(object) and isinstance(object, MetaData)


def get_regressors(module) -> dict[str, MetaData]:
    metadata_classes = getmembers(module, ismetadata)
    avail_regressors: dict[str, MetaData] = {}

    for _, metadata in metadata_classes:
        if getmodule(metadata) == module:
            regressor_name = metadata.name
            avail_regressors[regressor_name] = metadata

    return avail_regressors


def select_params(
    trial: optuna.Trial,
    potential_params: dict[str, list[Any] | NumberParams],
) -> dict[str, Any]:
    selected_params: dict[str, Any] = {}
    for param_name, params in potential_params.items():
        if isinstance(params, list):
            suggested_param: Any = trial.suggest_categorical(param_name, params)
        elif isinstance(params, dict):
            low = params["low"]
            high = params["high"]

            step = None
            if "step" in params:
                step = params["step"]

            ty = params["ty"]

            if isinstance(ty, float):
                suggested_param = trial.suggest_float(
                    param_name, low, high, step=step, log=True
                )
            elif isinstance(ty, int):
                low = int(low)
                high = int(high)

                if step is None:
                    suggested_param = trial.suggest_int(param_name, low, high, log=True)
                else:
                    step = int(step)
                    suggested_param = trial.suggest_int(
                        param_name, low, high, step=step, log=True
                    )

        selected_params[param_name] = suggested_param

    return selected_params


def objective(trial: optuna.Trial) -> float:
    if DATASET is None or TARGET is None:
        raise ValueError("Dataset or Target are not provided.")

    X: pd.DataFrame = DATASET.drop(columns=[TARGET])
    y: pd.Series = DATASET[TARGET]

    avail_regressors = get_regressors(regressors)

    regressor_name = trial.suggest_categorical(
        "regressor", list(avail_regressors.keys())
    )

    regressor_metadata = avail_regressors[regressor_name]
    potential_params = regressor_metadata.param_map
    selected_params = select_params(trial, potential_params)
    regressor_init = regressor_metadata.regressor(selected_params)

    score: np.ndarray = cross_val_score(regressor_init, X, y)
    return score.mean()


DATASET: pd.DataFrame | None = None
TARGET: str | None = None


def setup(dataset: pd.DataFrame, target: str):
    DATASET = dataset  # noqa: F841
    TARGET = target  # noqa: F841

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)
