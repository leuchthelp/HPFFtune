from pathlib import Path

import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier


def run(data: Path):

    clf = HistGradientBoostingClassifier().fit()
    clf.score()
