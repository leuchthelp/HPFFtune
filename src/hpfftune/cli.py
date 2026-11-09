import logging
from pathlib import Path

import pandas as pd
import xarray as xr
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

from hpfftune.learn.setup import setup

logging.basicConfig(
    level=logging.CRITICAL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)

error_console = Console(stderr=True)
install_rich_traceback(console=error_console)
logger.setLevel(logging.INFO)


def cli(ds_path: Path | None = None, target: str | None = None):
    ds_path = Path("sample-data/results.nc")
    target = "throughput_all_combined"

    dt: xr.DataTree = xr.open_datatree(ds_path)

    tmp = dt["overview"]

    if isinstance(tmp, xr.DataTree):
        overview: xr.DataTree = tmp
    else:
        raise TypeError("Somehow got array instead of subtree")

    combined: list[pd.DataFrame] = []
    for i in overview.descendants:
        for k in i.descendants:
            ds = k.to_dataset()
            combined.append(ds.to_dataframe())

    tmp = pd.concat(combined)
    test_nodes: pd.Series = tmp["on_node"]

    def change_type(x: str) -> int | None:
        try:
            tmp = str(x).split("l")
            return int(tmp[1])
        except IndexError:
            return None

    res = test_nodes.map(change_type)
    res = res.rename("node_num")
    tmp["node_num"] = res

    setup(tmp, target)
