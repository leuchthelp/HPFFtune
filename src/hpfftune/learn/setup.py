from inspect import getmembers, getmodule, isfunction
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from hpfftune.learn import regressors


def wrapped_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_test_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, **train_test_params)


def get_regressors(
    module, common_params: dict[str, Any]
) -> list[tuple[str, BaseEstimator]]:
    regressor_names = getmembers(module, isfunction)
    avail_regressors: list[tuple[str, BaseEstimator]] = []

    for _, func in regressor_names:
        if getmodule(func) == module:
            regressor: tuple[str, BaseEstimator] = func(common_params)
            avail_regressors.append(regressor)

    return avail_regressors


def setup(
    dataset: pd.DataFrame,
    target: str,
    common_params: dict[str, Any],
    train_test_params: dict[str, Any] | None = None,
    skip_hyper_params_tuning: bool = False,
    skip_dataset_cleanup: bool = False,
):
    X: pd.DataFrame = dataset.drop(columns=[target])
    y: pd.Series = dataset[target]

    if train_test_params is None:
        train_test_params = {}

    X_train, X_test, y_train, y_test = wrapped_train_test_split(X, y, train_test_params)

    avail_regressors = get_regressors(regressors, common_params)
