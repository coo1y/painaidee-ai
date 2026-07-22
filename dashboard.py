"""PaiNaiDee-AI — Monitoring dashboard.

Reads the SQLite interaction log and renders KPIs plus 6 charts:
  1. Positive vs negative feedback (donut)
  2. Daily query volume (line)
  3. Top provinces recommended (bar)
  4. Route usage: KB vs KB+Web (pie)
  5. Response latency distribution (histogram)
  6. Common feedback keywords (bar)
Plus a recent-comments table with sentiment highlighting.

Run with:  streamlit run dashboard.py
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from src.db import fetch_df

st.set_page_config(page_title="PaiNaiDee-AI — Monitoring", page_icon="📊", layout="wide")
st.title("📊 PaiNaiDee-AI — Monitoring Dashboard")

df = fetch_df()

if df.empty:
    st.info("No interactions logged yet. Chat with the app first (`streamlit run app.py`).")
    st.stop()

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
    st.subheader("1 · Feedback ratio")
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
    st.subheader("2 · Daily query volume")
    daily = df.assign(day=df["ts"].dt.date).groupby("day").size().reset_index(name="queries")
    fig = px.line(daily, x="day", y="queries", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Row 2: top provinces + route usage
# --------------------------------------------------------------------------
r2c1, r2c2 = st.columns(2)

with r2c1:
    st.subheader("3 · Top provinces recommended")
    prov = (
        df["provinces"].dropna().str.split(",").explode().str.strip().replace("", pd.NA).dropna()
    )
    if len(prov):
        top = prov.value_counts().head(10).reset_index()
        top.columns = ["province", "count"]
        fig = px.bar(top, x="count", y="province", orientation="h")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No province data yet.")

with r2c2:
    st.subheader("4 · Route usage")
    route_counts = df["route"].fillna("kb").value_counts().reset_index()
    route_counts.columns = ["route", "count"]
    fig = px.pie(route_counts, names="route", values="count", hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Row 3: latency + keywords
# --------------------------------------------------------------------------
r3c1, r3c2 = st.columns(2)

with r3c1:
    st.subheader("5 · Response latency (ms)")
    fig = px.histogram(df, x="latency_ms", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

_STOP = set(
    "the a an and or to of in for on is are i you it this that with my me we our can "
    "was were be do does answer response like dislike good bad not no yes very really "
    "because why what how when where too but so".split()
)

with r3c2:
    st.subheader("6 · Common feedback keywords")
    comments = df["comment"].dropna()
    words = []
    for c in comments:
        words += [w for w in re.findall(r"[a-zA-Z]{3,}", str(c).lower()) if w not in _STOP]
    if words:
        common = Counter(words).most_common(12)
        kdf = pd.DataFrame(common, columns=["keyword", "count"])
        fig = px.bar(kdf, x="count", y="keyword", orientation="h")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No text comments yet.")

# --------------------------------------------------------------------------
# Recent comments table with sentiment highlight
# --------------------------------------------------------------------------
st.divider()
st.subheader("💬 Recent user comments")
commented = df[df["comment"].notna() & (df["comment"].astype(str).str.len() > 0)]
if commented.empty:
    st.caption("No qualitative comments yet.")
else:
    view = commented[["ts", "query", "comment", "rating"]].head(20).copy()
    view["sentiment"] = view["rating"].map({1: "👍 Positive", 0: "👎 Negative"}).fillna("—")

    def _row_style(row):
        color = "#e8f8f0" if row["rating"] == 1 else ("#fdecea" if row["rating"] == 0 else "")
        return [f"background-color: {color}"] * len(row)

    styled = view.drop(columns=["rating"]).style.apply(
        lambda r: _row_style(view.loc[r.name]), axis=1
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
