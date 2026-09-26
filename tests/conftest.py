"""Shared pytest fixtures: put src/ on sys.path so tests import pairs_trading."""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src")
)
