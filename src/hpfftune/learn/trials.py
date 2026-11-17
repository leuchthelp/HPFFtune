import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.study import StudyDirection
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
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

        self.y_test = None
        if "test" in data:
            self.y_test = data["test"]

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

    def __call__(self, trial: optuna.Trial) -> list[float]:
        regressor_name = trial.suggest_categorical(
            "regressor", list(self.regressors.keys())
        )

        regressor_metadata = self.regressors[regressor_name]
        selected_params = regressor_metadata.param_map(trial)
        regressor_init = regressor_metadata.regressor(selected_params)

        X, y = regressor_metadata.clean_data(
            self.X, self.y if self.y_test is None else self.y_test
        )
        pipe = self.gen_pipeline(regressor_init, regressor_name, X)

        score: np.ndarray = cross_validate(
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

        _, _, pvalue = permutation_test_score(regressor_init, X, y, random_state=1)
        res.append(pvalue)

        return res


def create_study_metric_trial(
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
            n_trials=50,
            n_jobs=n_jobs if n_jobs else 1,
            show_progress_bar=True,
        )
    except ValueError:
        pass
