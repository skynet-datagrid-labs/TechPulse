from __future__ import annotations

"""Tests for the Streamlit dashboard module."""


def test_streamlit_app_imports() -> None:
    """Ensure the Streamlit app imports without runtime errors."""
    import importlib

    import streamlit_app

    importlib.reload(streamlit_app)
