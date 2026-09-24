import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from rich_argparse import RichHelpFormatter

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


def cli():

    pass
