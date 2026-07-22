"""PaiNaiDee-AI — multipage Streamlit entrypoint (deploy this on Streamlit Cloud).

Uses a sidebar selectbox to switch between page functions (same pattern as the
Streamlit demo gallery). Every page lives in ``views/`` and exposes ``render()``:
  1. 🧭 Chat        — views/chat.py
  2. 🧪 Evaluation  — views/evaluation.py
  3. 📊 Dashboard   — views/dashboard.py

On Streamlit Cloud set the "Main file path" to ``streamlit_app.py``.

Run locally with:  streamlit run streamlit_app.py
"""
from __future__ import annotations

# Non-GUI backend before any page imports wordcloud/matplotlib (avoids macOS
# GUI-backend segfaults when Streamlit runs page code off the main thread).
import matplotlib

matplotlib.use("Agg")

import streamlit as st

from src import config

# Never use server-side / .env API keys in the web UI — users enter their own.
config.scrub_server_api_keys()

# One global page config for the whole app. Must be the first Streamlit command.
st.set_page_config(
    page_title="PaiNaiDee-AI — Thailand Travel",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from views import chat, dashboard, evaluation

page_names_to_funcs = {
    "🧭 Chat": chat.render,
    "🧪 Evaluation": evaluation.render,
    "📊 Dashboard": dashboard.render,
}

page_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
page_names_to_funcs[page_name]()
