"""Run the dbt unit tests in DuckDB (transform/local_tests/run.py) as part of pytest, when DuckDB is installed."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("jinja2")


def test_dbt_unit_tests_pass_locally():
    runner = Path(__file__).resolve().parents[1] / "transform" / "local_tests" / "run.py"
    result = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
