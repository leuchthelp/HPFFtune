import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.study import StudyDirection
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    cross_validate,
)

from hpfftune.helpers import MetaData

logger = logging.getLogger(__name__)


class MetricTrial:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regressors: dict[str, MetaData],
        scoring: list[str],
        additional: list[str],
        cv: BaseCrossValidator | None = None,
    ):
        self.X = X
        self.y = y
        self.regressors = regressors

        self.scoring = scoring
        self.additional = additional
        self.cv = KFold(n_splits=5, shuffle=True) if cv is None else cv

    def __call__(self, trial: optuna.Trial) -> list[float]:
        regressor_name = trial.suggest_categorical(
            "regressor", list(self.regressors.keys())
        )

        regressor_metadata = self.regressors[regressor_name]
        selected_params = regressor_metadata.param_map(trial)
        regressor_init = regressor_metadata.regressor(selected_params)
        X, y = regressor_metadata.clean_data(self.X, self.y)

        score: np.ndarray = cross_validate(
            estimator=regressor_init,
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

        return res


def create_study_metric_trial(
    sampler: Any,
    regressor_meta: tuple[str, MetaData],
    scoring: dict[str, StudyDirection],
    additional: dict[str, StudyDirection],
) -> optuna.Study:
    study = optuna.create_study(
        sampler=sampler,
        directions=list(scoring.values()) + list(additional.values()),
        storage="sqlite:///newdb.sqlite3",
        study_name=f"{regressor_meta[0]}-{list(scoring.keys())}-{list(additional.keys())}",
        load_if_exists=True,
    )
    return study


def run_study_metric_trial(
    args: tuple[
        tuple[
            int,
            tuple[str, MetaData],
            dict[str, StudyDirection],
            dict[str, StudyDirection],
            tuple[pd.DataFrame, pd.Series],
        ],
        optuna.Study,
    ],
) -> None:
    n_jobs: int | None = args[0][0]
    regressor_meta: tuple[str, MetaData] = args[0][1]
    scoring: dict[str, StudyDirection] = args[0][2]
    additional: dict[str, StudyDirection] = args[0][3]
    data: tuple[pd.DataFrame, pd.Series] = args[0][4]
    study: optuna.Study = args[1]

    try:
        study.optimize(
            MetricTrial(
                data[0],
                data[1],
                {regressor_meta[0]: regressor_meta[1]},
                list(scoring.keys()),
                list(additional.keys()),
            ),
            n_trials=50,
            n_jobs=n_jobs if n_jobs else 1,
            show_progress_bar=True,
        )
    except ValueError:
        pass
