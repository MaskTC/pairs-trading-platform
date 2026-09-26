"""Tests for scripts/run_pipeline.py argument parsing and custom data loading."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import run_pipeline  # noqa: E402


def test_parse_args_defaults():
    args = run_pipeline.parse_args([])
    assert args.data == "yfinance"
    assert args.intraday_data == "yfinance"


def test_parse_args_custom_data_requires_path():
    with pytest.raises(SystemExit):
        run_pipeline.parse_args(["--data", "csv"])


def test_parse_args_custom_data_ok(tmp_path):
    p = tmp_path / "prices.csv"
    p.write_text("datetime,symbol,close\n2026-01-02,AAA,100.0\n")
    args = run_pipeline.parse_args(
        ["--data", "csv", "--data-path", str(p), "--data-format", "long"]
    )
    assert args.data_path == str(p)
    assert args.data_format == "long"


def test_load_custom_long(tmp_path):
    p = tmp_path / "prices.csv"
    p.write_text(
        "datetime,symbol,close\n"
        "2026-01-02,AAA,100.0\n2026-01-03,AAA,101.0\n"
        "2026-01-02,BBB,50.0\n2026-01-03,BBB,49.5\n"
    )
    out = run_pipeline._load_custom(str(p), "long")
    assert set(out.columns) == {"AAA", "BBB"}
    assert len(out) == 2


def test_load_custom_bad_format():
    with pytest.raises(SystemExit):
        run_pipeline._load_custom("/tmp/x.csv", "parquet")
