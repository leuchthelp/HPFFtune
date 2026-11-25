import importlib
import logging
import os
import time
from collections.abc import Callable
from inspect import getmembers, isclass
from itertools import product

import optuna
import optunahub
import pandas as pd
from optuna.study import StudyDirection
from pathos.multiprocessing import ProcessPool

from hpfftune.helpers import Data, MetaData
from hpfftune.learn import regressors, samplers
from hpfftune.learn.trials import (
    create_study_trial,
    run_study_metric_trial,
    run_study_sampler_comb_trial,
    run_study_sampler_trial,
)

logger = logging.getLogger(__name__)


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


def get_metadata(module) -> dict[str, MetaData]:
    metadata_classes = get_classes(module)
    avail: dict[str, MetaData] = {}

    for name in metadata_classes:
        metadata_init: MetaData = import_class(name, module.__name__)()
        regressor_name = metadata_init.name
        avail[regressor_name] = metadata_init

    return avail


def preprocess(
    X: pd.DataFrame,
    categorical_columns_subset: list[str],
    numerical_columns_subset: list[str],
) -> pd.DataFrame:
    X: pd.DataFrame = X[categorical_columns_subset + numerical_columns_subset]
    X[categorical_columns_subset] = X[categorical_columns_subset].astype("category")
    return X


def setup(
    dataset: pd.DataFrame,
    target: str,
    categorical_columns_subset: list[str],
    numerical_columns_subset: list[str],
    just_export: bool = False,
):
    X: pd.DataFrame = dataset.drop([target])
    y: pd.Series = dataset[target]

    X = preprocess(X, categorical_columns_subset, numerical_columns_subset)

    X_without_100GB: pd.DataFrame = dataset[dataset["total_filesize"] != 100].drop(
        [target]
    )
    y_without_100GB: pd.Series = dataset[dataset["total_filesize"] != 100][target]

    X_without_100GB = preprocess(
        X_without_100GB,
        categorical_columns_subset,
        numerical_columns_subset,
    )

    X_with_100GB: pd.DataFrame = dataset[dataset["total_filesize"] == 100].drop(
        [target]
    )
    X_with_100GB = preprocess(
        X_with_100GB,
        categorical_columns_subset,
        numerical_columns_subset,
    )

    y_with_100GB: pd.Series = dataset[dataset["total_filesize"] == 100][target]

    datasets: list[Data] = [
        {"train": (X, y), "path": "sqlite:///v2_metrics.sqlite3"},
        {
            "train": (X_without_100GB, y_without_100GB),
            "test": (X_with_100GB, y_with_100GB),
            "path": "sqlite:///v2_removed_100GB_data.sqlite3",
        },
    ]

    sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler(seed=10)
    avail_regressors = get_metadata(regressors)

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
        product([sampler], avail_regressors.items(), [scorings], [timings], datasets)
    )

    studies: list[optuna.Study] = []
    for trial in metric_trial_args:
        studies.append(
            create_study_trial(trial[0], trial[1], trial[2], trial[3], trial[4])
        )

    if just_export is False:
        processes = 4
        cpu_count = os.cpu_count()  # noqa: F841
        n_jobs = (
            None  # if cpu_count is None else (cpu_count - processes - 1) / processes
        )
        metric_trial_args = product(
            [n_jobs],
            avail_regressors.items(),
            [scorings],
            [timings],
            datasets,
        )

        metric_trial_args = list(zip(metric_trial_args, studies))

        pool = ProcessPool(processes=processes)
        logger.info(f"Running Metric trials with the following directives: {scorings}")
        start = time.monotonic()
        for _ in pool.uimap(run_study_metric_trial, metric_trial_args):
            pass
        logger.info(f"Finished Metric trials in {time.monotonic() - start} seconds")
    else:
        df_studies = [study.trials_dataframe() for study in studies]
        tmp = pd.concat(df_studies)

        new_names = list(scorings.keys()) + list(timings.keys()) + ["pvalue"]

        renameables: dict[str, str] = {}

        for idx, name in enumerate(new_names):
            renameables[f"values_{idx}"] = name

        tmp = tmp.rename(columns=renameables)
        tmp.to_json("sample-data/trials.json", orient="records")


def sampler_test(
    dataset: pd.DataFrame,
    target: str,
    categorical_columns_subset: list[str],
    numerical_columns_subset: list[str],
    just_export: bool = False,
) -> None:

    numerical_columns_subset.append(target)
    X = preprocess(dataset, categorical_columns_subset, numerical_columns_subset)

    X_100_samples = (
        X.reset_index(drop=True).drop_duplicates().sample(1000).drop(columns=[target])
    )
    y: pd.Series = X.reset_index(drop=True)[target]

    # datasets: list[Data] = [
    #     {
    #         "train": (X, pd.Series()),
    #         "path": "sqlite:///samplers.sqlite3",
    #     },
    # ]

    # sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler(seed=10)
    # avail_samplers = get_metadata(samplers)

    # scorings: dict[str, StudyDirection] = {
    #     target: StudyDirection.MAXIMIZE,
    # }
    # sampler_trial_args = list(
    #     product([sampler], avail_samplers.items(), [scorings], [{}], datasets)
    # )

    # studies: list[optuna.Study] = []
    # for trial in sampler_trial_args:
    #     studies.append(
    #         create_study_trial(trial[0], trial[1], trial[2], trial[3], trial[4])
    #     )

    # if just_export is False:
    #     processes = 4
    #     cpu_count = os.cpu_count()
    #     n_jobs = (
    #         None  # if cpu_count is None else (cpu_count - processes - 1) / processes
    #     )
    #     sampler_trial_args = product(
    #         [n_jobs],
    #         avail_samplers.items(),
    #         [scorings],
    #         [{}],
    #         datasets,
    #     )

    #     sampler_trial_args = list(zip(sampler_trial_args, studies))

    #     pool = ProcessPool(processes=processes)
    #     logger.info(f"Running Sampler trials with the following directives: {scorings}")
    #     start = time.monotonic()
    #     for _ in pool.uimap(run_study_sampler_trial, sampler_trial_args):
    #         pass
    #     logger.info(f"Finished Sampler trials in {time.monotonic() - start} seconds")
    # else:
    #     df_studies = [study.trials_dataframe() for study in studies]
    #     tmp = pd.concat(df_studies)

    #     new_names = list(scorings.keys())

    #     renameables: dict[str, str] = {}

    #     for idx, name in enumerate(new_names):
    #         renameables[f"values_{idx}"] = name

    #     tmp = tmp.rename(columns=renameables)
    #     tmp.to_json("sample-data/sampler-trials.json", orient="records")

    #
    #
    # Ugly but should work
    #
    #

    datasets: list[Data] = [
        {
            "train": (X.drop(columns=[target]), y),
            "path": "sqlite:///samplers_comb.sqlite3",
            "test": (X_100_samples, pd.Series()),
        },
    ]

    sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler(seed=10)
    avail_samplers = get_metadata(regressors)
    avail_samplers: dict[str, MetaData] = {
        "hist_grad_boost_reg": avail_samplers["hist_grad_boost_reg"]
    }

    scorings: dict[str, StudyDirection] = {
        target: StudyDirection.MAXIMIZE,
    }
    sampler_trial_args = list(
        product([sampler], avail_samplers.items(), [scorings], [{}], datasets)
    )

    studies: list[optuna.Study] = []
    for trial in sampler_trial_args:
        studies.append(
            create_study_trial(trial[0], trial[1], trial[2], trial[3], trial[4])
        )

    if just_export is False:
        processes = 4
        cpu_count = os.cpu_count()  # noqa: F841
        n_jobs = (
            None  # if cpu_count is None else (cpu_count - processes - 1) / processes
        )
        sampler_trial_args = product(
            [n_jobs],
            avail_samplers.items(),
            [scorings],
            [{}],
            datasets,
        )

        sampler_trial_args = list(zip(sampler_trial_args, studies))

        pool = ProcessPool(processes=processes)
        logger.info(f"Running Sampler trials with the following directives: {scorings}")
        start = time.monotonic()
        for _ in pool.uimap(run_study_sampler_comb_trial, sampler_trial_args):
            pass
        logger.info(f"Finished Sampler trials in {time.monotonic() - start} seconds")
    else:
        df_studies = [study.trials_dataframe() for study in studies]
        tmp = pd.concat(df_studies)

        new_names = list(scorings.keys())

        renameables: dict[str, str] = {}

        for idx, name in enumerate(new_names):
            renameables[f"values_{idx}"] = name

        tmp = tmp.rename(columns=renameables)
        tmp.to_json("sample-data/sampler-trials.json", orient="records")
