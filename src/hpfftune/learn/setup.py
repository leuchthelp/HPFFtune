import importlib
import logging
import os
import time
from collections.abc import Callable
from copy import deepcopy
from inspect import getmembers, isclass
from itertools import product
from typing import Any

import optuna
import optunahub
import pandas as pd
from optuna.study import StudyDirection
from pathos.multiprocessing import ProcessPool
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer

from hpfftune.helpers import Data, MetaData
from hpfftune.learn import regressors
from hpfftune.learn.trials import (
    create_study_metric_trial,
    run_study_metric_trial,
)

logger = logging.getLogger(__name__)


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


def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns_subset: list[str],
    numerical_columns_subset: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
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

    scaler = PowerTransformer()
    X: pd.DataFrame = pd.DataFrame(scaler.fit_transform(X))

    return X, y


def setup(
    dataset: pd.DataFrame,
    target: str,
    categorical_columns_subset: list[str],
    numerical_columns_subset: list[str],
):
    X: pd.DataFrame = dataset.drop([target])
    y: pd.Series = dataset[target]

    X, y = preprocess(X, y, categorical_columns_subset, numerical_columns_subset)

    X_without_100GB: pd.DataFrame = dataset[dataset["total_filesize"] != 100].drop(
        [target]
    )
    y_without_100GB: pd.Series = dataset[dataset["total_filesize"] != 100][target]

    X_without_100GB, y_without_100GB = preprocess(
        X_without_100GB,
        y_without_100GB,
        categorical_columns_subset,
        numerical_columns_subset,
    )

    X_with_100GB: pd.DataFrame = dataset[dataset["total_filesize"] == 100].drop(
        [target]
    )
    y_with_100GB: pd.Series = dataset[dataset["total_filesize"] == 100][target]

    X_with_100GB, y_with_100GB = preprocess(
        X_with_100GB,
        y_with_100GB,
        categorical_columns_subset,
        numerical_columns_subset,
    )

    datasets: list[Data] = [
        {"train": (X, y)},
        {
            "train": (X_without_100GB, y_without_100GB),
            "test": (X_with_100GB, y_with_100GB),
        },
    ]
    paths = ["sqlite:///metrics.sqlite3", "sqlite:///removed_100GB_data.sqlite3"]

    sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler(seed=10)
    avail_regressors = get_regressor_metadata(regressors)

    processes = 2
    pool = ProcessPool(processes=processes)

    cpu_count = os.cpu_count()

    scorings: dict[str, StudyDirection] = {
        "r2": StudyDirection.MAXIMIZE,
        "neg_mean_squared_error": StudyDirection.MAXIMIZE,
        "neg_mean_absolute_error": StudyDirection.MAXIMIZE,
        "d2_absolute_error_score": StudyDirection.MAXIMIZE,
    }
    timings: dict[str, StudyDirection] = {
        "fit_time": StudyDirection.MINIMIZE,
        "score_time": StudyDirection.MINIMIZE,
    }

    metric_trial_args = list(
        product(
            [sampler], avail_regressors.items(), [scorings], [timings], paths, datasets
        )
    )

    studies: list[optuna.Study] = []
    for trial in metric_trial_args:
        studies.append(
            create_study_metric_trial(trial[0], trial[1], trial[2], trial[3], trial[4])
        )

    n_jobs = None #if cpu_count is None else (cpu_count - processes - 1) / processes
    metric_trial_args = product(
        [n_jobs],
        avail_regressors.items(),
        [scorings],
        [timings],
        datasets,
        paths,
    )

    metric_trial_args = list(zip(metric_trial_args, studies))

    logger.info(f"Running Metric trials with the following directives: {scorings}")
    start = time.monotonic()
    for _ in pool.uimap(run_study_metric_trial, metric_trial_args):
        pass
    logger.info(f"Finished Metric trials in {time.monotonic() - start} seconds")
