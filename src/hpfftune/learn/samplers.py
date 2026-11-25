from dataclasses import dataclass
from typing import Any

import optuna
import optunahub
from optuna.samplers import BaseSampler, TPESampler

from hpfftune.helpers import MetaData


@dataclass
class TPESamplerMetaData(MetaData):
    def __init__(self):
        self.name = "TPE"

    def param_map(self, trial: optuna.Trial) -> dict[str, Any]:
        return {}

    def sampler(self, params: dict[str, Any]) -> BaseSampler:
        return TPESampler(**params)


@dataclass
class SMAC3SamplerMetaData(MetaData):
    def __init__(self):
        self.name = "SMAC3"

    def param_map(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "search_space": {},
            "n_trials": None,
            "surrogate_model_type": trial.suggest_categorical(
                "surrogate_model_type", ["rf"]
            ),
        }

    def sampler(self, params: dict[str, Any]) -> BaseSampler:
        sampler = optunahub.load_module("samplers/smac_sampler").SMACSampler
        return sampler(**params)
