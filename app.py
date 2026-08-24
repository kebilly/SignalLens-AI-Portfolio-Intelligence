from __future__ import annotations

import logging

import streamlit as st

from portfolio_app.config import Settings, SettingsError
from portfolio_app.ui import initialize_state, render_application

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

st.set_page_config(
    page_title="SignalLens | Portfolio Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    initialize_state()
    try:
        settings: Settings | None = Settings.load()
    except (SettingsError, ValueError):
        settings = None
    render_application(settings)


if __name__ == "__main__":
    main()
