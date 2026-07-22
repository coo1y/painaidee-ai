"""PaiNaiDee-AI — Monitoring dashboard.

Reads the SQLite interaction log and renders KPIs plus 6 charts:
  1. Positive vs negative feedback (donut)
  2. Daily query volume (line)
  3. Top provinces recommended (word cloud)
  4. Common feedback keywords (word cloud)
  5. Response latency distribution (histogram)
  6. Average latency every 30 minutes over the last 3 hours (line)
Plus a recent-response table (every query + response) with sentiment highlighting.

This is the 📊 Dashboard page of the multipage app. Run the whole app with:
  streamlit run streamlit_app.py
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

from src.db import fetch_df

# Thai-capable font (bundled) so word clouds render Thai province names instead
# of empty "tofu" rectangles. Falls back to the default font if missing.
_THAI_FONT = Path(__file__).resolve().parent / "assets" / "fonts" / "Sarabun-Regular.ttf"
_FONT_PATH = str(_THAI_FONT) if _THAI_FONT.exists() else None


def render_wordcloud(frequencies: dict[str, float], colormap: str = "viridis"):
    """Render a word cloud image from a {term: weight} mapping, or None if empty."""
    frequencies = {str(k): float(v) for k, v in frequencies.items() if k and v > 0}
    if not frequencies:
        return None
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=colormap,
        prefer_horizontal=0.9,
        font_path=_FONT_PATH,
        regexp=r"\S+",  # keep multi-char tokens whole (don't split Thai text)
    )
    return wc.generate_from_frequencies(frequencies).to_image()


# NOTE: st.set_page_config is intentionally NOT called here. Page config is set
# once in the multipage entrypoint (streamlit_app.py).
st.title("📊 PaiNaiDee-AI — Monitoring Dashboard")

df = fetch_df()

if df.empty:
    st.info("No interactions logged yet. Chat with the app first (`streamlit run app.py`).")
    st.stop()

# --------------------------------------------------------------------------
# Filters (top of page) — applied to the whole dashboard
# --------------------------------------------------------------------------
_SENTIMENT_LABELS = {1: "👍 Positive", 0: "👎 Negative"}


def _sentiment_label(rating) -> str:
    return _SENTIMENT_LABELS.get(rating, "— No rating")


df["sentiment_label"] = df["rating"].map(_sentiment_label)

f1, f2, f3 = st.columns([2, 2, 1])

with f1:
    valid_ts = df["ts"].dropna()
    min_date = valid_ts.min().date() if not valid_ts.empty else None
    max_date = valid_ts.max().date() if not valid_ts.empty else None
    date_range = None
    if min_date and max_date:
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

with f2:
    _options = ["👍 Positive", "👎 Negative", "— No rating"]
    selected_sentiments = st.multiselect("Sentiment", options=_options, default=_options)

with f3:
    st.write("")  # spacer to vertically align the button with the inputs
    st.write("")
    # Reload the latest app-usage data from the feedback DB (fetch_df reads fresh).
    if st.button("🔄 Refresh data", use_container_width=True):
        st.rerun()

# Apply the date filter.
if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    _start, _end = date_range
    _day = df["ts"].dt.date
    df = df[(_day >= _start) & (_day <= _end)]

# Apply the sentiment filter.
if selected_sentiments:
    df = df[df["sentiment_label"].isin(selected_sentiments)]

st.caption(f"Showing **{len(df)}** interaction(s) after filters.")

if df.empty:
    st.warning("No interactions match the current filters. Adjust the filters above.")
    st.stop()

st.divider()

# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
rated = df[df["rating"].notna()]
pos = int((rated["rating"] == 1).sum())
neg = int((rated["rating"] == 0).sum())
sat = (pos / (pos + neg) * 100) if (pos + neg) else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total queries", len(df))
c2.metric("Rated responses", len(rated))
c3.metric("Satisfaction", f"{sat:.0f}%")
c4.metric("Avg latency", f"{df['latency_ms'].mean():.0f} ms")

st.divider()

# --------------------------------------------------------------------------
# Row 1: feedback ratio + daily volume
# --------------------------------------------------------------------------
r1c1, r1c2 = st.columns(2)

with r1c1:
    st.subheader("1.) Feedback ratio")
    if pos + neg:
        fig = px.pie(
            names=["👍 Positive", "👎 Negative"],
            values=[pos, neg],
            hole=0.55,
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No ratings collected yet.")

with r1c2:
    st.subheader("2.) Daily query volume")
    daily = df.assign(day=df["ts"].dt.date).groupby("day").size().reset_index(name="queries")
    daily["day"] = daily["day"].astype(str)  # discrete category labels on the X-axis
    fig = px.bar(daily, x="day", y="queries", text="queries")
    fig.update_xaxes(type="category", title="Date")
    fig.update_yaxes(title="Queries")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Row 2: top provinces + feedback keywords
# --------------------------------------------------------------------------
r2c1, r2c2 = st.columns(2)

with r2c1:
    st.subheader("3.) Top provinces recommended")
    prov = (
        df["provinces"].dropna().str.split(",").explode().str.strip().replace("", pd.NA).dropna()
    )
    if len(prov):
        img = render_wordcloud(prov.value_counts().to_dict(), colormap="viridis")
        st.image(img, width="stretch")
    else:
        st.caption("No province data yet.")

_STOP = set(
    "the a an and or to of in for on is are i you it this that with my me we our can "
    "was were be do does answer response like dislike good bad not no yes very really "
    "because why what how when where too but so".split()
)

with r2c2:
    st.subheader("4.) Common feedback keywords")
    comments = df["comment"].dropna()
    words = []
    for c in comments:
        words += [w for w in re.findall(r"[a-zA-Z]{3,}", str(c).lower()) if w not in _STOP]
    if words:
        img = render_wordcloud(dict(Counter(words)), colormap="plasma")
        st.image(img, width="stretch")
    else:
        st.caption("No text comments yet.")

# --------------------------------------------------------------------------
# Row 3: latency distribution + last-3h latency
# --------------------------------------------------------------------------
r3c1, r3c2 = st.columns(2)

with r3c1:
    st.subheader("5.) Response latency (ms)")
    fig = px.histogram(df, x="latency_ms", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

with r3c2:
    st.subheader("6.) Latency — last 3 hours")
    lat = df.dropna(subset=["latency_ms"]).copy()
    window_end = pd.Timestamp.now(tz="UTC")
    window_start = window_end - pd.Timedelta(hours=3)
    lat = lat[(lat["ts"] >= window_start) & (lat["ts"] <= window_end)]
    if not lat.empty:
        lat["time_bucket"] = lat["ts"].dt.floor("30min")
        interval_lat = (
            lat.groupby("time_bucket")["latency_ms"]
            .mean()
            .round(0)
            .reset_index(name="avg_latency_ms")
        )
        fig = px.line(
            interval_lat,
            x="time_bucket",
            y="avg_latency_ms",
            markers=True,
        )
        fig.update_xaxes(
            title="Time (30-minute intervals)",
            tickformat="%H:%M",
            dtick=30 * 60 * 1000,
            range=[window_start, window_end],
        )
        fig.update_yaxes(title="Avg latency (ms)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No latency data in the last 3 hours.")

# --------------------------------------------------------------------------
# Recent responses table (every query + response, with sentiment highlight)
# --------------------------------------------------------------------------
st.divider()
st.subheader("💬 Recent responses")
if df.empty:
    st.caption("No responses yet.")
else:
    view = df.sort_values("ts", ascending=False).head(50)[
        ["ts", "query", "response", "comment", "rating"]
    ].copy()
    view["comment"] = view["comment"].fillna("")
    view["sentiment"] = view["rating"].map({1: "👍 Positive", 0: "👎 Negative"}).fillna("—")

    display = view.drop(columns=["rating"])[
        ["ts", "query", "response", "comment", "sentiment"]
    ]

    def _row_style(row):
        # `row` comes from `display`; look up the rating from `view` by index.
        rating = view.loc[row.name, "rating"]
        color = "#e8f8f0" if rating == 1 else ("#fdecea" if rating == 0 else "")
        return [f"background-color: {color}"] * len(row)

    styled = display.style.apply(_row_style, axis=1)
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ts": st.column_config.DatetimeColumn("Time", width="small"),
            "query": st.column_config.TextColumn("User query", width="medium"),
            "response": st.column_config.TextColumn("AI response", width="large"),
            "comment": st.column_config.TextColumn("Comment", width="medium"),
            "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
        },
    )
