"""Rich console output and logging setup.

Results go to stdout (``console``), so they can be piped or redirected.
Logs go to stderr (``err_console``).
"""

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()
err_console = Console(stderr=True)


def setup_logging(level: int = logging.INFO) -> None:
    """Send all log records through a Rich handler on stderr."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=err_console, rich_tracebacks=True)],
        force=True,
    )
