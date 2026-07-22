# PaiNaiDee-AI — Agentic RAG Travel Recommender for Thailand

PaiNaiDee-AI ("ไปไหนดี" = *"where should I go?"*) is an end-to-end **Agentic RAG**
application for planning trips in Thailand. It grounds answers in official
Tourism Authority of Thailand (TAT) data — **attractions** and **events** — and
optionally calls live **web search** when the knowledge base cannot cover
real-time facts. Answers are in English, with source citations.

This README is written for reviewers who did **not** take the course. It explains
the problem, the data, the application flow, how to run everything, and where
each **evaluation criterion** is satisfied. See
[Evaluation criteria mapping](#evaluation-criteria-mapping).

---

## Table of contents

- [Problem description](#problem-description)
- [The data](#the-data)
- [Architecture and flow](#architecture-and-flow)
- [Project structure](#project-structure)
- [Quickstart (local)](#quickstart-local)
- [Quickstart (Docker)](#quickstart-docker)
- [Usage examples](#usage-examples)
- [Ingestion pipeline](#ingestion-pipeline)
- [Retrieval and best practices](#retrieval-and-best-practices)
- [Evaluation](#evaluation)
- [Interface](#interface)
- [Monitoring and feedback](#monitoring-and-feedback)
- [Containerization](#containerization)
- [Deployment (Streamlit Cloud)](#deployment-streamlit-cloud)
- [Configuration reference](#configuration-reference)
- [Evaluation criteria mapping](#evaluation-criteria-mapping)
- [Reproducibility checklist](#reproducibility-checklist)

---

## Problem description

**Score target: 2/2** — clear problem and solution.

Planning a trip in Thailand means combining scattered information: *what* to see,
*where* it is, *when* events happen, and practical details (hours, contacts,
prices). Official TAT exports are rich but hard to query: Thai-heavy narrative
fields, HTML markup, free-text opening hours, and time-bound events mixed with
evergreen attractions. General chatbots often invent opening times and dates.

**PaiNaiDee-AI solves this** by grounding an LLM in TAT data:

- Ask in plain English (e.g. *"cultural riverside towns in eastern Thailand"* or
  *"festivals near Trat around New Year"*).
- The app retrieves relevant **attractions** and **events**, filters events by
  **province** and **date range**, re-ranks candidates, and returns a **cited**
  answer.
- For facts missing from the KB (live ticket prices, weather, transport), a
  **web-search tool** (Tavily) can run.
- Every answer can be rated 👍/👎 with an optional comment; usage feeds a
  **monitoring dashboard**.

The system is designed **not to invent** missing facts — it states uncertainty
and cites `[S#]` (knowledge base) or `[W#]` (web) sources.

---

## The data

JSON exports from the **Tourism Authority of Thailand (TAT)** live in `data/`:

| File | Entity | Approx. size | Key fields |
|------|--------|--------------|------------|
| `attraction.json` | Static attractions | ~8.6k records | `ATT_NAME_TH/EN`, `ATT_DETAIL_TH` (HTML), `ATT_HILIGHT`, `ATT_LOCATION` (`lat, lng`), `PROVINCE_NAME_TH`, `ATT_CATEGORY_LABEL`, `ATT_START_END`, contacts |
| `activity.json` | Scheduled events | ~20k records | `NAME`, `DESCRIPTION` (HTML), `STARTDATE`, `ENDDATE`, `PROVINCE`, `LOCATION`, `EVENTTARGETGROUP`, `TATEVENTTYPENAME`, ticket prices, contacts |

Optional small samples (`attraction-sub.json`, `activity-sub.json`) remain for
schema demos. The pipeline defaults to the full files via `src/config.py`
(`ATTRACTION_FILE` / `ACTIVITY_FILE`).

**Structure note:** each export is a single-key JSON object whose key is the
original SQL query and whose value is the row array. Ingestion unwraps this
automatically (`src/utils.py` → `load_tat_export`).

**Cleaning handled by the pipeline:** HTML stripping, bilingual text, `"lat, lng"`
parsing, Oracle `LISTAGG` trailing commas, ISO dates + epoch-day metadata for
event filtering, and duplicate-ID / batch upserts for large corpora.

---

## Architecture and flow

```
   data/*.json  ──►  Prefect ingestion (clean · normalize · embed · metadata)
                              │
                              ▼
                    ChromaDB (attractions | events)
                              │
 user query ──► Query rewriting ──► Hybrid search (dense + BM25, RRF)
                                      + metadata filter (province, dates)
                                      ▼
                                   LLM re-ranking
                                      │
                         Router ──────┤ need real-time / weak KB?
                            │         ▼
                            │    Tavily web search (optional)
                            ▼
              Grounded prompt ──► OpenAI LLM ──► cited English answer
                                                      │
                                                      ▼
                               SQLite log + 👍/👎 feedback ──► Dashboard
```

**Stack:** Streamlit (multipage UI) · OpenAI (chat + embeddings) · ChromaDB ·
rank-bm25 · Tavily · Prefect · SQLite · Docker Compose · wordcloud / Plotly
(dashboard).

---

## Project structure

```text
.
├── data/
│   ├── attraction.json          # full TAT attractions
│   ├── activity.json            # full TAT events
│   ├── attraction-sub.json      # tiny sample (optional)
│   └── activity-sub.json
├── assets/fonts/
│   └── Sarabun-Regular.ttf      # Thai-capable font for word clouds
├── ingestion/
│   └── ingest_pipeline.py       # Prefect flow: clean → embed → ChromaDB
├── src/
│   ├── config.py                # env-driven configuration
│   ├── utils.py                 # HTML strip, coord/date parse
│   ├── embeddings.py            # OpenAI (+ offline local) embeddings
│   ├── llm.py                   # OpenAI chat wrapper
│   ├── retrieval.py             # hybrid search, rewrite, filter, rerank
│   ├── agent.py                 # router + Tavily + answer synthesis
│   └── db.py                    # SQLite interaction / feedback store
├── eval/
│   ├── retrieval_eval.py        # vector vs bm25 vs hybrid vs hybrid+rr
│   ├── llm_eval.py              # prompts A / B / C + LLM judge
│   └── test_data/
│       ├── ground_truth.json    # 100 labeled retrieval cases
│       └── questions.json       # LLM-eval question set
├── views/
│   ├── chat.py                  # Chat page (UI + feedback)
│   ├── evaluation.py            # Evaluation results page
│   └── dashboard.py             # Monitoring dashboard (6+ charts)
├── streamlit_app.py             # multipage entrypoint (Chat · Eval · Dashboard)
├── app.py                       # chat UI (standalone)
├── dashboard.py                 # dashboard (standalone)
├── Dockerfile
├── docker-compose.yml           # ingest → app → dashboard
├── requirements.txt             # pinned dependency versions
├── .env.example
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
└── README.md
```

---

## Quickstart (local)

**Prerequisites:** Python 3.10+.

```bash
# 1. Clone and enter
git clone <your-repo-url> && cd PaiNaiDee-AI

# 2. Virtual env + pinned dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Secrets
cp .env.example .env
# Edit .env: OPENAI_API_KEY (required), TAVILY_API_KEY (optional)

# 4. Ingest into ChromaDB (Prefect-orchestrated)
python -m ingestion.ingest_pipeline

# 5. Multipage app (Chat · Evaluation · Dashboard)
streamlit run streamlit_app.py   # → http://localhost:8501
```

Standalone alternatives:

```bash
streamlit run app.py             # chat only
streamlit run dashboard.py       # dashboard only
```

**No OpenAI key?** Set `EMBEDDING_BACKEND=local` for an offline smoke test
(hash embeddings + template answers). Use a real key for production quality.

---

## Quickstart (Docker)

Full stack via Docker Compose (ingestion → Streamlit app → dashboard), with
shared volumes for Chroma and SQLite:

```bash
cp .env.example .env     # add your keys
docker compose up --build
```

| Service | URL |
|---------|-----|
| Chat app (`app.py`) | http://localhost:8501 |
| Dashboard (`dashboard.py`) | http://localhost:8502 |

The `ingest` service runs once and must succeed before `app` starts.

---

## Usage examples

| You ask | What happens |
|---------|--------------|
| *"Recommend a cultural riverside old town in eastern Thailand."* | Query rewrite → hybrid retrieval on attractions → re-rank → cited answer. |
| *"Any festivals in Trat around New Year?"* | Rewriter extracts province + date window → event metadata filter → cited event. |
| *"How much is a ticket to the Phuket Boat Show right now?"* | Router may trigger **Tavily**; answer blends KB + web with `[W#]` citations. |

Each reply shows a **Sources** expander, a route caption (KB vs KB+Web), and
👍/👎 feedback with an optional comment.

---

## Ingestion pipeline

**Score target: 2/2** — automated ingestion with a dedicated tool (**Prefect**).

`ingestion/ingest_pipeline.py` defines flow `painaidee-ingestion`:

1. **Load & normalize** — unwrap SQL-keyed JSON, strip HTML (BeautifulSoup),
   parse coordinates/dates, build bilingual documents, attach filterable
   metadata.
2. **Build collections** — embed with OpenAI `text-embedding-3-small` (or local
   backend), upsert into Chroma collections `attractions` and `events` (batched,
   de-duplicated by ID).

```bash
python -m ingestion.ingest_pipeline          # reset + rebuild
python -m ingestion.ingest_pipeline --no-reset
```

`run_ingestion()` is a Prefect-free path used by the Streamlit app for cold-start
auto-ingest (useful on Streamlit Cloud).

---

## Retrieval and best practices

Implemented in `src/retrieval.py` and used by `src/agent.py`:

| Best practice | Implementation | Criterion |
|---------------|----------------|-----------|
| **Hybrid search** | Dense (Chroma) + BM25 (`rank-bm25`), fused with Reciprocal Rank Fusion | +1 |
| **Document re-ranking** | LLM scores candidates 0–10 and reorders top-k | +1 |
| **User query rewriting** | LLM rewrites search terms and extracts province / date / source filters | +1 |

Also included: **metadata filtering** (event date-range overlap + province) with
fallback if a strict filter returns empty.

The live app default is **query rewrite → hybrid search → re-ranking**, which
won the retrieval evaluation (`hybrid+rr`).

The router in `src/agent.py` decides whether to call **Tavily**, then synthesizes
a grounded English answer with `[S#]` / `[W#]` citations.

---

## Evaluation

**Score targets:** Retrieval evaluation **2/2**, LLM evaluation **2/2**
(multiple approaches compared; best approach used in the app).

### Retrieval evaluation

```bash
python -m eval.retrieval_eval
```

Compares four methods on `eval/test_data/ground_truth.json` (**100** cases:
50 attractions + 50 events):

| Method | Recall@3 | MRR@3 |
|--------|----------|-------|
| vector | 0.200 | 0.180 |
| bm25 | 0.500 | 0.408 |
| hybrid | 0.460 | 0.302 |
| **hybrid+rr** | **0.620** | **0.605** |

**Best approach: `hybrid+rr`** — used as the app default.

Results are also shown on the **🧪 Evaluation** page (`views/evaluation.py`).

### LLM evaluation

```bash
python -m eval.llm_eval    # requires OPENAI_API_KEY
```

Compares **three** system prompts on the same retrieved context, scored by an
LLM judge (faithfulness + helpfulness, 1–5). Questions live in
`eval/test_data/questions.json`.

| Variant | Faithfulness | Helpfulness | Overall |
|---------|--------------|-------------|---------|
| A · concise | 4.90 | 4.67 | 4.79 |
| B · detailed planner | 4.81 | 4.67 | 4.74 |
| **C · production** | 4.81 | **4.90** | **4.86** |

**Best prompt: C · production** — this is the system prompt in `src/agent.py`
(`_ANSWER_SYS`).

> Large question sets are expensive (many sequential OpenAI calls). Prefer a
> modest `questions.json` for routine runs; the Evaluation page stores the
> reported numbers for demos without re-running the API.

---

## Interface

**Score target: 2/2** — Streamlit UI.

Main entrypoint: `streamlit run streamlit_app.py`

| Page | Module | Role |
|------|--------|------|
| 🧭 Chat | `views/chat.py` | Conversational recommender + feedback |
| 🧪 Evaluation | `views/evaluation.py` | Retrieval + LLM eval tables / charts |
| 📊 Dashboard | `views/dashboard.py` | Monitoring KPIs and charts |

---

## Monitoring and feedback

**Score target: 2/2** — user feedback **and** a dashboard with **≥5 charts**.

### Feedback collection

In Chat, each answer supports:

- Streamlit `st.feedback` 👍 / 👎
- Optional free-text comment

Stored in SQLite (`feedback.db`) via `src/db.py`: query, response, route,
provinces, latency, rating, comment, timestamp.

### Dashboard (≥6 visuals)

`views/dashboard.py` / `dashboard.py`:

1. Positive vs negative feedback (donut)
2. Daily query volume (vertical bar chart, date on X-axis)
3. Top provinces recommended (**word cloud**, Thai font)
4. Common feedback keywords (**word cloud**)
5. Response latency distribution (histogram)
6. Average latency — last 3 hours, 30-minute buckets (line)

Plus:

- Top filters: **date range** + **sentiment**
- **Refresh data** button
- **Recent response** table (every logged query + AI answer, not only commented rows)

---

## Containerization

**Score target: 2/2** — full `docker-compose` for the stack.

| Artifact | Role |
|----------|------|
| `Dockerfile` | Streamlit application image |
| `docker-compose.yml` | `ingest` → `app` → `dashboard` with shared volumes |

See [Quickstart (Docker)](#quickstart-docker).

---

## Deployment (Streamlit Cloud)

**Bonus target: +2** — cloud deployment.

The app uses **embedded ChromaDB** (no separate DB server):

1. Push the repo to GitHub.
2. Create a Streamlit Cloud app with **Main file path = `streamlit_app.py`**.
3. Add secrets (see `.streamlit/secrets.toml.example`):

   ```toml
   OPENAI_API_KEY = "sk-..."
   TAVILY_API_KEY = "tvly-..."   # optional
   ```

4. Deploy. On first launch the app can auto-ingest via `run_ingestion()`.

**Cold start note:** embedding ~29k documents on first Cloud boot is slow/costly.
Prefer a pre-built index workflow or a trimmed dataset for demos. The Evaluation
page uses precomputed numbers so it does not call the eval APIs live.

---

## Configuration reference

Environment variables (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required for embeddings + chat |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` (example) | Chat model (override in `.env`) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `TAVILY_API_KEY` | — | Enables web search |
| `EMBEDDING_BACKEND` | `openai` | `openai` or `local` (offline test) |
| `CHROMA_DIR` | `./chroma_db` | Vector store path |
| `FEEDBACK_DB` | `./feedback.db` | SQLite feedback DB |
| `DATA_DIR` | `./data` | Source JSON directory |

Dependency versions are pinned in `requirements.txt`.

---

## Evaluation criteria mapping

| Criterion | Points | Where in this project |
|-----------|--------|------------------------|
| **Problem description** | 2 | [Problem description](#problem-description) |
| **Retrieval flow** (KB + LLM) | 2 | `src/retrieval.py`, `src/agent.py`, ChromaDB |
| **Retrieval evaluation** (multiple approaches; best used) | 2 | `eval/retrieval_eval.py`, `eval/test_data/ground_truth.json`; app uses **hybrid+rr** |
| **LLM evaluation** (multiple approaches; best used) | 2 | `eval/llm_eval.py` (prompts A/B/C); app uses **prompt C** in `src/agent.py` |
| **Interface** | 2 | Streamlit multipage UI — `streamlit_app.py`, `views/` |
| **Ingestion pipeline** | 2 | Prefect flow — `ingestion/ingest_pipeline.py` |
| **Monitoring** (feedback + ≥5 charts) | 2 | Chat feedback + `views/dashboard.py` (6 charts + table) |
| **Containerization** | 2 | `Dockerfile` + `docker-compose.yml` (ingest, app, dashboard) |
| **Reproducibility** | 2 | This README, pinned `requirements.txt`, accessible `data/` |
| **Hybrid search** | +1 | `src/retrieval.py` (`hybrid_search`) + retrieval eval |
| **Document re-ranking** | +1 | `src/retrieval.py` (`rerank`) + retrieval eval |
| **User query rewriting** | +1 | `src/retrieval.py` (`rewrite_query`) |
| **Bonus: cloud deployment** | +2 | [Streamlit Cloud](#deployment-streamlit-cloud) |

---

## Reproducibility checklist

1. Clone the repo and use Python 3.10+.
2. `pip install -r requirements.txt` (versions pinned).
3. Copy `.env.example` → `.env` and set API keys.
4. Ensure `data/attraction.json` and `data/activity.json` are present.
5. Run `python -m ingestion.ingest_pipeline`.
6. Run `streamlit run streamlit_app.py`.
7. Optional quality checks:
   - `python -m eval.retrieval_eval`
   - `python -m eval.llm_eval`
8. Or: `docker compose up --build`.

---

## Data attribution

Tourism data © **Tourism Authority of Thailand (TAT)**. Included for
educational / demo purposes.
