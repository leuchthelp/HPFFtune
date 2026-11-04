from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from hpfftune.helpers import get_calling_func_name


def hist_grad_boost_reg(
    common_params: dict[str, Any],
) -> tuple[str, BaseEstimator]:
    regressor = HistGradientBoostingRegressor(**common_params)
    return get_calling_func_name(), regressor


def random_forest_reg(common_params: dict[str, Any]) -> tuple[str, BaseEstimator]:
    regressor = RandomForestRegressor(**common_params)
    return get_calling_func_name(), regressor
