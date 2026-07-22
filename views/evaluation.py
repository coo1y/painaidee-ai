"""🧪 Evaluation — LLM & retrieval results (filled in manually).

This page reports the offline evaluation results produced by:
  • eval/retrieval_eval.py  →  Recall@k / MRR@k for 4 retrieval approaches
  • eval/llm_eval.py        →  faithfulness / helpfulness for 3 prompt variants

Streamlit Cloud can't run the full evaluations on every visit (they need the
OpenAI API and the full index), so the numbers below are entered by hand after
running the scripts locally:

    python -m eval.retrieval_eval
    python -m eval.llm_eval

>>> EDIT THE VALUES IN THE TWO DICTS BELOW after each evaluation run. <<<

Called via ``evaluation.render()`` from ``streamlit_app.py``.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================================================
# 1) RETRIEVAL EVALUATION — fill in from `python -m eval.retrieval_eval`
# ==========================================================================
RETRIEVAL_META = {
    "k": 3,
    "embeddings": "openai",
    "ground_truth_cases": 100,
}

# method -> {recall@k, mrr@k}
RETRIEVAL_RESULTS = {
    "vector": {"recall@k": 0.200, "mrr@k": 0.180},
    "bm25": {"recall@k": 0.500, "mrr@k": 0.408},
    "hybrid": {"recall@k": 0.460, "mrr@k": 0.302},
    "hybrid+rr": {"recall@k": 0.620, "mrr@k": 0.605},
}
RETRIEVAL_BEST = "hybrid+rr"

# ==========================================================================
# 2) LLM EVALUATION — fill in from `python -m eval.llm_eval`
# ==========================================================================
LLM_META = {
    "questions_evaluated": 21,
    "judge_model": "gpt-4o-mini",
}

# variant -> {faithfulness, helpfulness}  (1-5 scale)
LLM_RESULTS = {
    "A · concise": {"faithfulness": 4.90, "helpfulness": 4.67},
    "B · detailed planner": {"faithfulness": 4.81, "helpfulness": 4.67},
    "C · production": {"faithfulness": 4.81, "helpfulness": 4.90},
}
LLM_BEST = "C · production"

# System prompts from eval/llm_eval.py (PROMPT_A / B / C)
LLM_PROMPTS = {
    "A · concise": (
        "You are PaiNaiDee, a Thailand travel assistant. Answer in English, concisely "
        "(2-4 sentences). Use ONLY the provided context and cite sources as [S#]. "
        "Do not invent details."
    ),
    "B · detailed planner": (
        "You are PaiNaiDee, an expert Thailand trip planner. Answer in English with a "
        "short intro then bullet points (name, why to go, practical tip). Use ONLY the "
        "provided context and cite sources as [S#]. If a detail is missing, say so."
    ),
    "C · production": (
        "You are PaiNaiDee, a friendly Thailand travel assistant. Answer in English. "
        "Use ONLY the provided context and cite every claim with its tag, e.g. [S1]. "
        "If the context lacks specific details (exact price, current hours, "
        "availability), say so honestly instead of inventing them. Recommend concrete "
        "places/events and briefly explain why. Be concise and practical."
    ),
}


def _retrieval_df() -> pd.DataFrame:
    return pd.DataFrame(RETRIEVAL_RESULTS).T.reset_index(names="method")


def _llm_df() -> pd.DataFrame:
    df = pd.DataFrame(LLM_RESULTS).T.reset_index(names="variant")
    df["overall"] = (df["faithfulness"] + df["helpfulness"]) / 2
    return df


def render() -> None:
    """Render the evaluation page (called by streamlit_app.py)."""
    st.title("🧪 Evaluation results")
    st.caption(
        "Offline quality metrics for the retrieval layer and the answer-generation "
        "prompt. Numbers are entered manually after running the eval scripts locally."
    )

    _has_retrieval = any(
        v["recall@k"] or v["mrr@k"] for v in RETRIEVAL_RESULTS.values()
    )
    _has_llm = any(
        v["faithfulness"] or v["helpfulness"] for v in LLM_RESULTS.values()
    )
    if not (_has_retrieval or _has_llm):
        st.info(
            "These are placeholder values. Run `python -m eval.retrieval_eval` and "
            "`python -m eval.llm_eval`, then paste the numbers into "
            "`views/evaluation.py`.",
            icon="✏️",
        )

    st.divider()

    st.header("1) Retrieval evaluation")
    st.caption(
        f"Recall@{RETRIEVAL_META['k']} and MRR@{RETRIEVAL_META['k']} · "
        f"embeddings = `{RETRIEVAL_META['embeddings']}` · "
        f"{RETRIEVAL_META['ground_truth_cases']:,} ground-truth cases"
    )

    r_df = _retrieval_df()

    rc1, rc2 = st.columns([3, 2])
    with rc1:
        st.dataframe(
            r_df.style.format({"recall@k": "{:.3f}", "mrr@k": "{:.3f}"}).highlight_max(
                subset=["recall@k", "mrr@k"], color="#e8f8f0"
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "method": st.column_config.TextColumn("Method"),
                "recall@k": st.column_config.NumberColumn(
                    f"Recall@{RETRIEVAL_META['k']}"
                ),
                "mrr@k": st.column_config.NumberColumn(f"MRR@{RETRIEVAL_META['k']}"),
            },
        )
        st.success(
            f"**Best approach: {RETRIEVAL_BEST}** — the default used by the app."
        )

    with rc2:
        melted = r_df.melt(id_vars="method", var_name="metric", value_name="score")
        fig = px.bar(
            melted,
            x="method",
            y="score",
            color="metric",
            barmode="group",
            color_discrete_sequence=["#2e86de", "#54a0ff"],
        )
        fig.update_layout(xaxis_title=None, yaxis_title="Score", legend_title=None)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.header("2) LLM answer evaluation")
    st.caption(
        f"Prompt variants scored 1–5 by an LLM judge on "
        f"**faithfulness** (grounded in context) and **helpfulness** · "
        f"{LLM_META['questions_evaluated']} questions · "
        f"judge = `{LLM_META['judge_model']}`"
    )

    l_df = _llm_df()

    lc1, lc2 = st.columns([3, 2])
    with lc1:
        st.dataframe(
            l_df.style.format(
                {
                    "faithfulness": "{:.2f}",
                    "helpfulness": "{:.2f}",
                    "overall": "{:.2f}",
                }
            ).highlight_max(subset=["overall"], color="#e8f8f0"),
            hide_index=True,
            use_container_width=True,
            column_config={
                "variant": st.column_config.TextColumn("Prompt variant"),
                "faithfulness": st.column_config.NumberColumn("Faithfulness"),
                "helpfulness": st.column_config.NumberColumn("Helpfulness"),
                "overall": st.column_config.NumberColumn("Overall"),
            },
        )
        st.success(
            f"**Best prompt: {LLM_BEST}** — mirrors the prompt in `src/agent.py`."
        )

    with lc2:
        melted = l_df.melt(
            id_vars="variant",
            value_vars=["faithfulness", "helpfulness"],
            var_name="metric",
            value_name="score",
        )
        fig = px.bar(
            melted,
            x="variant",
            y="score",
            color="metric",
            barmode="group",
            range_y=[0, 5],
            color_discrete_sequence=["#10ac84", "#1dd1a1"],
        )
        fig.update_layout(
            xaxis_title=None, yaxis_title="Score (1–5)", legend_title=None
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prompt variants")
    for name, prompt in LLM_PROMPTS.items():
        with st.expander(name, expanded=(name == LLM_BEST)):
            st.markdown(f"```\n{prompt}\n```")

    st.divider()
    st.caption(
        "To refresh: run the eval scripts locally, then update `RETRIEVAL_RESULTS` "
        "and `LLM_RESULTS` in `views/evaluation.py` and redeploy."
    )
