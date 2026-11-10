from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.study import StudyDirection
from sklearn.model_selection import BaseCrossValidator, RepeatedKFold, cross_validate

from hpfftune.helpers import MetaData


class MetricTrial:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regressors: dict[str, MetaData],
        scoring: str,
        cv: BaseCrossValidator | None = None,
    ):
        self.X = X
        self.y = y
        self.regressors = regressors

        self.scoring = scoring
        self.cv = RepeatedKFold(n_splits=5) if cv is None else cv

    def __call__(self, trial: optuna.Trial) -> float:
        regressor_name = trial.suggest_categorical(
            "regressor", list(self.regressors.keys())
        )

        regressor_metadata = self.regressors[regressor_name]
        selected_params = regressor_metadata.param_map(trial)
        regressor_init = regressor_metadata.regressor(selected_params)
        X = regressor_metadata.clean_data(self.X)

        score: np.ndarray = cross_validate(
            estimator=regressor_init,
            X=X,
            y=self.y,
            cv=self.cv,
            scoring=[self.scoring],
        )
        return score[f"test_{self.scoring}"].mean()


def create_study_metric_trial(
    sampler: Any,
    regressor_meta: tuple[str, MetaData],
    scoring: tuple[str, StudyDirection],
) -> optuna.Study:
    study = optuna.create_study(
        sampler=sampler,
        direction=scoring[1],
        storage="sqlite:///db.sqlite3",
        study_name=f"MetricTrial-{scoring[0]}-{regressor_meta[0]}",
        load_if_exists=True,
    )
    return study


def run_study_metric_trial(
    args: tuple[
        tuple[
            int,
            tuple[str, MetaData],
            tuple[str, StudyDirection],
            tuple[pd.DataFrame, pd.Series],
        ],
        optuna.Study,
    ],
) -> None:
    n_jobs: int | None = args[0][0]
    regressor_meta: tuple[str, MetaData] = args[0][1]
    scoring: tuple[str, StudyDirection] = args[0][2]
    data: tuple[pd.DataFrame, pd.Series] = args[0][3]
    study: optuna.Study = args[1]

    study.optimize(
        MetricTrial(
            data[0], data[1], {regressor_meta[0]: regressor_meta[1]}, scoring[0]
        ),
        n_trials=100,
        n_jobs=n_jobs if n_jobs else 1,
        show_progress_bar=True,
    )


class TimeTrial:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regressors: dict[str, MetaData],
        timing: str,
        cv: BaseCrossValidator | None = None,
    ):
        self.X = X
        self.y = y
        self.regressors = regressors

        self.timing = timing
        self.cv = RepeatedKFold(n_splits=5) if cv is None else cv

    def __call__(self, trial: optuna.Trial) -> float:
        regressor_name = trial.suggest_categorical(
            "regressor", list(self.regressors.keys())
        )

        regressor_metadata = self.regressors[regressor_name]
        selected_params = regressor_metadata.param_map(trial)
        regressor_init = regressor_metadata.regressor(selected_params)
        X = regressor_metadata.clean_data(self.X)

        score: np.ndarray = cross_validate(
            estimator=regressor_init,
            X=X,
            y=self.y,
            cv=self.cv,
        )
        return score[self.timing].mean()


def create_study_time_trial(
    sampler: Any,
    regressor_meta: tuple[str, MetaData],
    timing: tuple[str, StudyDirection],
) -> optuna.Study:
    study = optuna.create_study(
        sampler=sampler,
        direction=timing[1],
        storage="sqlite:///db.sqlite3",
        study_name=f"TimeTrial-{timing[0]}-{regressor_meta[0]}",
        load_if_exists=True,
    )
    return study


def run_study_time_trial(
    args: tuple[
        tuple[
            int,
            tuple[str, MetaData],
            tuple[str, StudyDirection],
            tuple[pd.DataFrame, pd.Series],
        ],
        optuna.Study,
    ],
) -> None:
    n_jobs: int | None = args[0][0]
    regressor_meta: tuple[str, MetaData] = args[0][1]
    timing: tuple[str, StudyDirection] = args[0][2]
    data: tuple[pd.DataFrame, pd.Series] = args[0][3]
    study: optuna.Study = args[1]

    study.optimize(
        TimeTrial(data[0], data[1], {regressor_meta[0]: regressor_meta[1]}, timing[0]),
        n_trials=100,
        n_jobs=n_jobs if n_jobs else 1,
        show_progress_bar=True,
    )
