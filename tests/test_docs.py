"""The generated reference pages of the docs are up to date."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _generator():  # noqa: ANN202 - a module loaded from a file
    spec = importlib.util.spec_from_file_location(
        "gen_reference", ROOT / "scripts" / "gen_reference.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("page", ["config.md", "cli.md"])
def test_reference_pages_are_current(page: str) -> None:
    generator = _generator()
    expected = generator.PAGES[page]()
    actual = (ROOT / "docs" / "reference" / page).read_text()
    assert actual == expected, (
        f"docs/reference/{page} is out of date: run "
        "`uv run python scripts/gen_reference.py`"
    )
