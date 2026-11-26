import logging
import time
from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.study import StudyDirection
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    cross_validate,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer

from hpfftune.helpers import Data, MetaData

logger = logging.getLogger(__name__)


def create_study_trial(
    sampler: Any,
    regressor_meta: tuple[str, MetaData],
    scoring: dict[str, StudyDirection],
    additional: dict[str, StudyDirection],
    data: Data,
) -> optuna.Study:
    name = data["path"].split("/")[-1].split(".")[0]

    study = optuna.create_study(
        sampler=sampler,
        directions=list(scoring.values())
        + list(additional.values())
        + [StudyDirection.MINIMIZE],
        storage=data["path"],
        study_name=f"{name}-{regressor_meta[0]}-{list(scoring.keys())}-{list(additional.keys())}-[p_value]",
        load_if_exists=True,
    )
    return study


class MetricTrial:
    def __init__(
        self,
        data: Data,
        regressors: dict[str, MetaData],
        scoring: list[str],
        additional: list[str],
        cv: BaseCrossValidator | None = None,
    ):
        self.X, self.y = data["train"]
        self.regressors = regressors

        self.X_test, self.y_test = None, None
        if "test" in data:
            self.X_test, self.y_test = data["test"]

        self.scoring = scoring
        self.additional = additional
        self.cv = KFold(n_splits=5, shuffle=True, random_state=1) if cv is None else cv

    def gen_pipeline(
        self, estimator: BaseEstimator, name: str, X: pd.DataFrame
    ) -> Pipeline:
        columns_to_encode: list[str] = X.select_dtypes(
            exclude=["float", "int"]
        ).columns.to_list()

        return Pipeline(
            [
                (
                    "column_transformer",
                    ColumnTransformer(
                        transformers=[("oe", OrdinalEncoder(), columns_to_encode)],
                        remainder="passthrough",
                    ),
                ),
                ("power_transformer", PowerTransformer()),
                (name, estimator),
            ]
        )

    def apply_custom_validate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        pipe: Pipeline,
    ) -> dict[str, np.ndarray]:

        res: dict[str, np.ndarray] = {}
        score_time: list[float] = []

        start = time.monotonic()
        pipe: Pipeline = pipe.fit(X_train, y_train)
        fit_time: list[float] = [time.monotonic() - start]

        for name in self.scoring:
            scorer: Callable = get_scorer(name)

            start = time.monotonic()
            score: float = scorer(
                estimator=pipe,
                X=X_test,
                y_true=y_test,
            )
            score_time.append(time.monotonic() - start)
            res[f"test_{name}"] = np.array(score)

        res["fit_time"] = np.array(fit_time)
        res["score_time"] = np.array(score_time)
        return res

    def __call__(self, trial: optuna.Trial) -> list[float]:
        regressor_name = trial.suggest_categorical(
            "regressor", list(self.regressors.keys())
        )

        regressor_metadata = self.regressors[regressor_name]
        selected_params = regressor_metadata.param_map(trial)
        regressor_init = regressor_metadata.regressor(selected_params)

        if self.X_test is not None and self.y_test is not None:
            X, y = regressor_metadata.clean_data(self.X_test, self.y_test)
            X_train, y_train = regressor_metadata.clean_data(self.X, self.y)
            pipe = self.gen_pipeline(regressor_init, regressor_name, X)

            score: dict[str, np.ndarray] = self.apply_custom_validate(
                X, y, X_train, y_train, pipe
            )
        else:
            X, y = regressor_metadata.clean_data(self.X, self.y)
            pipe = self.gen_pipeline(regressor_init, regressor_name, X)

            score: dict[str, np.ndarray] = cross_validate(
                estimator=pipe,
                X=X,
                y=y,
                cv=self.cv,
                scoring=self.scoring,
            )

        res: list[float] = []
        for metric in self.scoring:
            res.append(score[f"test_{metric}"].mean())
        for metric in self.additional:
            res.append(score[metric].mean())

        _, _, pvalue = permutation_test_score(pipe, X, y, random_state=1)
        res.append(pvalue)

        return res


def run_study_metric_trial(
    args: tuple[
        tuple[
            int,
            tuple[str, MetaData],
            dict[str, StudyDirection],
            dict[str, StudyDirection],
            Data,
            str,
        ],
        optuna.Study,
    ],
) -> None:
    n_jobs: int | None = args[0][0]
    regressor_meta: tuple[str, MetaData] = args[0][1]
    scoring: dict[str, StudyDirection] = args[0][2]
    additional: dict[str, StudyDirection] = args[0][3]
    data: Data = args[0][4]
    study: optuna.Study = args[1]

    try:
        study.optimize(
            MetricTrial(
                data,
                {regressor_meta[0]: regressor_meta[1]},
                list(scoring.keys()),
                list(additional.keys()),
            ),
            n_trials=25,
            n_jobs=n_jobs if n_jobs else 1,
            show_progress_bar=True,
        )
    except ValueError:
        pass


class InnerSampleTrial:
    def __init__(
        self,
        X: pd.DataFrame,
        scoring: dict[str, StudyDirection],
    ):
        self.X = X

        self.scoring = scoring

    def __call__(self, trial: optuna.Trial) -> list[float]:
        parameters: dict[str, Any] = {
            "format": trial.suggest_categorical("format", ["zarr", "hdf5", "netcdf4"]),
            "task_type": trial.suggest_categorical("task_type", ["read"]),
            "language": trial.suggest_categorical("language", ["py"]),
            "no_caching": trial.suggest_categorical("no_caching", [True]),
            "parallel_backend": trial.suggest_categorical(
                "parallel_backend", ["MPI", None]
            ),
            "access_kind": trial.suggest_categorical("access_kind", ["independent"]),
            "unit_var": trial.suggest_categorical("unit_var", ["GB"]),
            "unit_chunk": trial.suggest_categorical("unit_chunk", ["MB"]),
            "dimensions": trial.suggest_int("dimensions", 1, 3, step=1),
            "num_nodes": trial.suggest_categorical("num_nodes", [1]),
            "num_ranks": trial.suggest_categorical("num_ranks", [1, 8, 16, 32, 64]),
            "parallel": trial.suggest_categorical("parallel", [True, False]),
            "total_filesize": trial.suggest_categorical(
                "total_filesize", [10, 50, 100]
            ),
            "filesize_per_var": trial.suggest_categorical(
                "filesize_per_var", [10, 50, 100]
            ),
            "filesize_per_chunk": trial.suggest_categorical(
                "filesize_per_chunk", [2, 4, 8, 16, 32, 64, 128]
            ),
        }

        values = self.X.reset_index(drop=True)
        for key, value in parameters.items():
            values: pd.DataFrame = values.loc[(values[key] == value)]

        res: list[float] = []
        for entry in self.scoring:
            r: float = 0

            df: pd.Series = values[entry]
            if not df.empty:
                r = float(df.mean())

            res.append(r)
        return res


class SamplerTrial:
    def __init__(
        self,
        data: Data,
        sampler: dict[str, MetaData],
        scoring: dict[str, StudyDirection],
    ):
        self.path = data["path"]
        self.X, self.y = data["train"]
        self.sampler = sampler

        self.X_test, self.y_test = None, None
        if "test" in data:
            self.X_test, self.y_test = data["test"]

        self.scoring = scoring
        self.i = 0

    def __call__(self, trial: optuna.Trial) -> list[float]:
        sampler_name = trial.suggest_categorical("sampler", list(self.sampler.keys()))

        sampler_metadata = self.sampler[sampler_name]
        selected_params = sampler_metadata.param_map(trial)

        parameters: dict[str, Any] = {
            "format": optuna.distributions.CategoricalDistribution(
                ["zarr", "hdf5", "netcdf4"]
            ),
            "task_type": optuna.distributions.CategoricalDistribution(["read"]),
            "language": optuna.distributions.CategoricalDistribution(["py"]),
            "no_caching": optuna.distributions.CategoricalDistribution([True]),
            "parallel_backend": optuna.distributions.CategoricalDistribution(
                ["MPI", None]
            ),
            "access_kind": optuna.distributions.CategoricalDistribution(
                ["independent"]
            ),
            "unit_var": optuna.distributions.CategoricalDistribution(["GB"]),
            "unit_chunk": optuna.distributions.CategoricalDistribution(["MB"]),
            "dimensions": optuna.distributions.IntDistribution(1, 3, step=1),
            "num_nodes": optuna.distributions.CategoricalDistribution([1]),
            "num_ranks": optuna.distributions.CategoricalDistribution(
                [1, 8, 16, 32, 64]
            ),
            "parallel": optuna.distributions.CategoricalDistribution([True, False]),
            "total_filesize": optuna.distributions.CategoricalDistribution(
                [10, 50, 100]
            ),
            "filesize_per_var": optuna.distributions.CategoricalDistribution(
                [10, 50, 100]
            ),
            "filesize_per_chunk": optuna.distributions.CategoricalDistribution(
                [2, 4, 8, 16, 32, 64, 128]
            ),
        }

        if "search_space" in selected_params:
            selected_params["search_space"] = parameters

        n_trials = 100
        if "n_trials" in selected_params:
            selected_params["n_trials"] = n_trials

        sampler_init = sampler_metadata.sampler(selected_params)
        study = optuna.create_study(
            sampler=sampler_init,
            directions=list(
                self.scoring.values(),
            ),
            study_name=f"InnerSamplerTrial{self.i}-{sampler_name}-{list(self.scoring.keys())}",
            storage=self.path,
            load_if_exists=True,
        )

        self.i += 1

        try:
            study.optimize(
                InnerSampleTrial(self.X, self.scoring),
                n_trials=n_trials,
                show_progress_bar=True,
            )

            return [study.best_value, 1]

        except ValueError:
            return [0, 1]


def run_study_sampler_trial(
    args: tuple[
        tuple[
            int,
            tuple[str, MetaData],
            dict[str, StudyDirection],
            dict[str, StudyDirection],
            Data,
            str,
        ],
        optuna.Study,
    ],
) -> None:
    n_jobs: int | None = args[0][0]
    sampler_meta: tuple[str, MetaData] = args[0][1]
    scoring: dict[str, StudyDirection] = args[0][2]
    data: Data = args[0][4]
    study: optuna.Study = args[1]

    try:
        study.optimize(
            SamplerTrial(
                data,
                {sampler_meta[0]: sampler_meta[1]},
                scoring,
            ),
            n_trials=5,
            n_jobs=n_jobs if n_jobs else 1,
            show_progress_bar=True,
        )
    except ValueError:
        pass


class SamplerCombTrial:
    def __init__(
        self,
        data: Data,
        regressors: dict[str, MetaData],
        scoring: list[str],
        cv: BaseCrossValidator | None = None,
    ):
        self.X, self.y = data["train"]
        self.regressors = regressors

        self.X_test, self.y_test = None, None
        if "test" in data:
            self.X_test, self.y_test = data["test"]

        self.scoring = scoring
        self.cv = KFold(n_splits=5, shuffle=True, random_state=1) if cv is None else cv

    def gen_pipeline(
        self, estimator: BaseEstimator, name: str, X: pd.DataFrame
    ) -> Pipeline:
        columns_to_encode: list[str] = X.select_dtypes(
            exclude=["float", "int"]
        ).columns.to_list()

        return Pipeline(
            [
                (
                    "column_transformer",
                    ColumnTransformer(
                        transformers=[("oe", OrdinalEncoder(), columns_to_encode)],
                        remainder="passthrough",
                    ),
                ),
                ("power_transformer", PowerTransformer()),
                (name, estimator),
            ]
        )

    def __call__(self, trial: optuna.Trial) -> list[float]:
        regressor_name = trial.suggest_categorical(
            "regressor", list(self.regressors.keys())
        )

        regressor_metadata = self.regressors[regressor_name]
        selected_params = regressor_metadata.param_map(trial)
        regressor_init = regressor_metadata.regressor(selected_params)

        X, y = regressor_metadata.clean_data(self.X, self.y)
        pipe = self.gen_pipeline(regressor_init, regressor_name, X)
        pipe.fit(X, y)

        pred: pd.Series = pd.Series(pipe.predict(self.X_test))
        idx = pred.idxmax()

        if self.X_test is None:
            raise ValueError

        print(f"config: {dict(self.X_test.iloc[idx])!s}")
        # trial.suggest_categorical("config", [str(dict(self.X_test.iloc[idx]))])
        return [pred.max(), 1]


def run_study_sampler_comb_trial(
    args: tuple[
        tuple[
            int,
            tuple[str, MetaData],
            dict[str, StudyDirection],
            dict[str, StudyDirection],
            Data,
            str,
        ],
        optuna.Study,
    ],
) -> None:
    n_jobs: int | None = args[0][0]
    regressor_meta: tuple[str, MetaData] = args[0][1]
    scoring: dict[str, StudyDirection] = args[0][2]
    data: Data = args[0][4]
    study: optuna.Study = args[1]

    try:
        study.optimize(
            SamplerCombTrial(
                data,
                {regressor_meta[0]: regressor_meta[1]},
                list(scoring.keys()),
            ),
            n_trials=5,
            n_jobs=n_jobs if n_jobs else 1,
            show_progress_bar=True,
        )
    except ValueError:
        pass
