import pytest

from hpfftune.helpers import MetaData, get_calling_func_name
from hpfftune.learn import regressors
from hpfftune.learn.regressors import (
    HistGradBoostRegMetaData,
    RandomForestRegressorMetaData,
)
from hpfftune.learn.setup import get_regressor_metadata


def test_default():
    assert True


@pytest.mark.parametrize(
    "input1, expectation",
    [
        (get_calling_func_name, "test_get_calling_func_name"),
    ],
)
def test_get_calling_func_name(input1, expectation):
    assert input1() == expectation


@pytest.mark.parametrize(
    "expectation",
    [
        {
            "hist_grad_boost_reg": HistGradBoostRegMetaData,
            "random_forest_reg": RandomForestRegressorMetaData,
        },
    ],
)
def test_get_regressors(expectation: dict[str, MetaData]):
    avail_regressors = get_regressor_metadata(regressors)

    assert avail_regressors

    print(avail_regressors)

    for name, metadata in avail_regressors.items():
        assert name in expectation
        known_regressor = expectation[name]
        assert type(metadata) == known_regressor
