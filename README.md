# 🧭 PaiNaiDee-AI — Agentic RAG Travel Recommender for Thailand

PaiNaiDee-AI ("ไปไหนดี" = *"where should I go?"*) is an end-to-end **Agentic
RAG** application that helps travelers plan trips in Thailand. It combines a
curated knowledge base of **attractions** and **events** (from the Tourism
Authority of Thailand) with **live web search**, and answers questions in
English with cited sources.

> Built as a course capstone. This README is written for readers who did **not**
> take the course — it explains the problem, the data, the flow, and how to run
> everything. The [Evaluation criteria](#-evaluation-criteria-mapping) section
> maps each rubric item to where it lives in the repo.

---

## 📋 Table of contents
- [Problem description](#-problem-description)
- [The data](#-the-data)
- [Architecture & flow](#-architecture--flow)
- [Project structure](#-project-structure)
- [Quickstart (local)](#-quickstart-local)
- [Quickstart (Docker)](#-quickstart-docker)
- [Usage examples](#-usage-examples)
- [Ingestion pipeline](#-ingestion-pipeline)
- [Retrieval & best practices](#-retrieval--best-practices)
- [Evaluation](#-evaluation)
- [Monitoring & feedback](#-monitoring--feedback)
- [Deployment (Streamlit Cloud)](#-deployment-streamlit-cloud)
- [Configuration reference](#-configuration-reference)
- [Evaluation criteria mapping](#-evaluation-criteria-mapping)

---

## 🎯 Problem description

Planning a trip in Thailand means juggling scattered information: *what* to see,
*where* it is, *when* events happen, and *practical* details (hours, contacts,
prices). Official tourism data is rich but stored in raw, Thai-language,
HTML-laden database exports that are hard to query in natural language. General
chatbots, meanwhile, hallucinate opening times and event dates.

**PaiNaiDee-AI solves this** by grounding an LLM in an official TAT dataset:

- Ask in plain English (e.g. *"cultural riverside towns in eastern Thailand"* or
  *"festivals near Trat around New Year"*).
- The app retrieves relevant **attractions** and **events**, filters events by
  **date range** and **province**, and generates a concise, **cited** answer.
- For **real-time** facts the KB can't cover (current ticket prices, weather,
  transport), a **web-search tool** is triggered automatically.
- Every answer can be rated 👍/👎 with a comment, feeding a **monitoring
  dashboard**.

The system is explicitly designed **not to invent** missing facts (prices,
hours, availability) — it says so honestly and cites what it used.

---

## 🗂 The data

Two JSON exports from the **Tourism Authority of Thailand (TAT)** live in
`data/`:

| File | Entity | Key fields |
|------|--------|-----------|
| `attraction-sub.json` | Static attractions | `ATT_NAME_TH/EN`, `ATT_DETAIL_TH` (HTML), `ATT_HILIGHT`, `ATT_LOCATION` (lat,lng), `PROVINCE_NAME_TH`, `ATT_CATEGORY_LABEL`, `ATT_START_END`, contacts |
| `activity-sub.json` | Scheduled events | `NAME`, `DESCRIPTION` (HTML), `STARTDATE`, `ENDDATE`, `PROVINCE`, `LOCATION`, `EVENTTARGETGROUP`, `TATEVENTTYPENAME`, ticket prices, contacts |

**Structure note:** each file is a single-key JSON object whose key is the
original SQL query and whose value is the list of rows. The ingestion pipeline
unwraps this automatically.

**Data characteristics handled by the pipeline:** HTML tags in narrative fields,
Thai + English bilingual text (events are often Thai-only), `"lat, lng"`
coordinate strings, Oracle `LISTAGG` artifacts (trailing `,`), and mixed
Buddhist/Gregorian year references.

The two files shipped here are **small samples** (3 records each) intended for
schema design and reproducible demos. The pipeline scales unchanged to the full
export — just replace the files.

---

## 🏛 Architecture & flow

```
                         ┌─────────────────────────────────────────────┐
   data/*.json  ──────►  │  Ingestion pipeline (Prefect)                │
                         │  clean HTML · normalize · embed · metadata   │
                         └───────────────┬─────────────────────────────┘
                                         ▼
                         ┌─────────────────────────────────────────────┐
                         │  ChromaDB (persistent, embedded)             │
                         │  collections: attractions | events           │
                         └───────────────┬─────────────────────────────┘
                                         ▼
 user query ─► Query rewriting ─► Hybrid search (dense + BM25, RRF)
                                     │      + metadata filter (province, dates)
                                     ▼
                                  Re-ranking (LLM cross-scoring)
                                     │
                    Router ──────────┤ needs real-time / weak coverage?
                       │             ▼
                       │        Tavily web search  ── (optional) ──┐
                       ▼                                           ▼
                 Grounded prompt  ──►  OpenAI LLM  ──►  cited English answer
                                                            │
                                                            ▼
                                        SQLite log + 👍/👎 feedback ─► Dashboard
```

**Tech stack:** Streamlit (UI + dashboard) · OpenAI (LLM + embeddings) ·
ChromaDB (vector store) · rank-bm25 (lexical) · Tavily (web search) · Prefect
(ingestion orchestration) · SQLite (feedback) · Docker Compose.

---

## 📁 Project structure

```text
.
├── data/
│   ├── attraction-sub.json      # static attractions (TAT)
│   └── activity-sub.json        # scheduled events (TAT)
├── ingestion/
│   └── ingest_pipeline.py       # Prefect flow: clean → embed → load ChromaDB
├── src/
│   ├── config.py                # env-driven configuration
│   ├── utils.py                 # HTML strip, coord/date parse, normalization
│   ├── embeddings.py            # OpenAI (+ offline test) embeddings
│   ├── llm.py                   # OpenAI chat wrapper (+ JSON mode)
│   ├── retrieval.py             # hybrid search, query rewrite, metadata filter, rerank
│   ├── agent.py                 # router agent + Tavily web-search tool + synthesis
│   └── db.py                    # SQLite interaction log & feedback
├── eval/
│   ├── retrieval_eval.py        # vector vs bm25 vs hybrid vs hybrid+rerank
│   └── llm_eval.py              # compare 2 prompts with an LLM judge
├── app.py                       # Streamlit chat UI + feedback
├── dashboard.py                 # monitoring dashboard (6 charts)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt             # pinned versions
├── .env.example
└── README.md
```

---

## 🚀 Quickstart (local)

**Prerequisites:** Python 3.10+.

```bash
# 1. Clone and enter
git clone <your-repo-url> && cd PaiNaiDee-AI

# 2. Create a virtual env and install pinned dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env
#   Edit .env and set OPENAI_API_KEY (required) and TAVILY_API_KEY (optional)

# 4. Ingest the data into ChromaDB (orchestrated with Prefect)
python -m ingestion.ingest_pipeline

# 5. Run the chat app
streamlit run app.py            # ➜ http://localhost:8501

# 6. (Optional) Run the monitoring dashboard in another terminal
streamlit run dashboard.py      # ➜ http://localhost:8502
```

> **No OpenAI key handy?** Set `EMBEDDING_BACKEND=local` in `.env` to run a fully
> **offline smoke test** (deterministic hash embeddings + a template answer).
> Retrieval still works via the BM25 channel. This mode is for wiring/tests only
> — set your real key for production-quality answers.

---

## 🐳 Quickstart (Docker)

Everything (ingestion → app → dashboard) is orchestrated by Docker Compose with
shared volumes for the Chroma index and the SQLite feedback DB.

```bash
cp .env.example .env     # add your keys
docker compose up --build
```

- Chat app: <http://localhost:8501>
- Dashboard: <http://localhost:8502>

The `ingest` service runs once and populates a named volume; `app` starts only
after ingestion completes successfully.

---

## 💬 Usage examples

| You ask | What happens |
|---------|--------------|
| *"Recommend a cultural riverside old town in eastern Thailand."* | Query rewritten → hybrid search on `attractions` → reranked → cited answer (e.g. Chanthaboon Waterfront Community). |
| *"Any festivals in Trat around New Year?"* | Rewriter extracts province `ตราด` + date window → event metadata filter → cited event answer with dates. |
| *"How much is a ticket to the Phuket Boat Show right now?"* | KB lacks live price → router triggers **Tavily** web search → answer blends KB + web with `[W#]` citations and a note that prices should be verified. |

Each assistant reply shows a **📚 Sources** expander, a route indicator (KB vs
KB+Web), and 👍/👎 buttons with an optional comment box.

---

## 🔧 Ingestion pipeline

`ingestion/ingest_pipeline.py` is a **Prefect** flow (`painaidee-ingestion`) with
tasks that:

1. **Load & normalize** — unwrap the SQL-keyed JSON, strip HTML
   (BeautifulSoup), parse `"lat,lng"`, normalize dates to ISO + epoch-days,
   split `LISTAGG` lists.
2. **Build collections** — embed documents (OpenAI `text-embedding-3-small`) and
   upsert into two ChromaDB collections, `attractions` and `events`, with
   searchable metadata (province, dates as epoch-days, coordinates, contacts).

Run it as an orchestrated flow:

```bash
python -m ingestion.ingest_pipeline          # reset + rebuild
python -m ingestion.ingest_pipeline --no-reset  # incremental upsert
```

A Prefect-free `run_ingestion()` function is also exposed and is used by the
Streamlit app to auto-build the KB on first launch (handy for Streamlit Cloud).

---

## 🔎 Retrieval & best practices

Implemented in `src/retrieval.py`:

- **Hybrid search** — dense vector search (Chroma) **+** BM25 lexical search
  (`rank-bm25`), fused with **Reciprocal Rank Fusion**.
- **Query rewriting** — an LLM rewrites the user question into optimized search
  terms and extracts structured filters (Thai province name, date window,
  source preference).
- **Metadata filtering** — events are pre-filtered by **date-range overlap** and
  **province** before semantic search, with a graceful fallback if a strict
  filter empties the result set.
- **Re-ranking** — an LLM cross-scores candidates 0–10 and reorders the top-k.

The router (`src/agent.py`) then decides whether to also call the **Tavily**
web-search tool (real-time facts or weak KB coverage), and synthesizes a grounded
answer that cites every claim as `[S#]` (knowledge base) or `[W#]` (web).

---

## 📊 Evaluation

### Retrieval evaluation
```bash
python -m eval.retrieval_eval
```
Compares **four** approaches — `vector`, `bm25`, `hybrid`, `hybrid+rerank` — on
Recall@k and MRR@k. Uses `eval/ground_truth.json` if present, otherwise
auto-generates a proxy set. On the small sample data these scores are a smoke
test; with the full dataset + curated queries they differentiate approaches. The
app ships with **hybrid + re-ranking** as the default.

### LLM evaluation
```bash
python -m eval.llm_eval      # requires OPENAI_API_KEY
```
Compares **two prompt variants** (concise vs. detailed planner) on the same
retrieved context, scored by an **LLM judge** on faithfulness and helpfulness.
The higher-scoring prompt is reported.

---

## 📈 Monitoring & feedback

- **Feedback collection** — every response has 👍/👎 buttons (Streamlit
  `st.feedback`) plus an optional free-text comment. All interactions (query,
  response, route, provinces, latency, rating, comment, timestamp) are logged to
  **SQLite** (`feedback.db`) via `src/db.py`.
- **Dashboard** (`dashboard.py`) — KPIs + **6 charts**:
  1. Positive vs negative feedback (donut)
  2. Daily query volume (line)
  3. Top provinces recommended (bar)
  4. Route usage: KB vs KB+Web (pie)
  5. Response latency distribution (histogram)
  6. Common feedback keywords (bar)
  - plus a recent-comments table with sentiment highlighting.

> _Add screenshots here_: `docs/app.png`, `docs/dashboard.png`. In Streamlit you
> can record a short preview video from the top-right menu and drag it into this
> README on GitHub.

---

## ☁️ Deployment (Streamlit Cloud)

The app uses **embedded ChromaDB** (no separate DB server), so it deploys to
Streamlit Cloud directly:

1. Push this repo to GitHub.
2. Create a new Streamlit Cloud app pointing at `app.py`.
3. In **App settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   TAVILY_API_KEY = "tvly-..."   # optional
   ```
4. Deploy. On first launch the app auto-ingests the data into a local Chroma
   store (`run_ingestion()`), so no manual step is required.

---

## ⚙️ Configuration reference

All settings are environment variables (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required for real embeddings + chat. |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat/completion model. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model. |
| `TAVILY_API_KEY` | — | Enables the web-search tool (skipped if empty). |
| `EMBEDDING_BACKEND` | `openai` | `openai` or `local` (offline test only). |
| `CHROMA_DIR` | `./chroma_db` | Vector store location. |
| `FEEDBACK_DB` | `./feedback.db` | SQLite feedback DB. |
| `DATA_DIR` | `./data` | Source JSON location. |

Dependency versions are pinned in `requirements.txt`.

---

## ✅ Evaluation criteria mapping

| Criterion | Where |
|-----------|-------|
| **Problem description** | [Problem description](#-problem-description) |
| **Retrieval flow** (KB + LLM) | `src/retrieval.py` + `src/agent.py` |
| **Retrieval evaluation** (multiple approaches) | `eval/retrieval_eval.py` (vector/bm25/hybrid/hybrid+rerank) |
| **LLM evaluation** (multiple approaches) | `eval/llm_eval.py` (2 prompts + judge) |
| **Interface** | `app.py` (Streamlit UI) |
| **Ingestion pipeline** (automated tool) | `ingestion/ingest_pipeline.py` (Prefect) |
| **Monitoring** (feedback + dashboard ≥5 charts) | `app.py` feedback + `dashboard.py` (6 charts) |
| **Containerization** (full docker-compose) | `Dockerfile`, `docker-compose.yml` |
| **Reproducibility** | This README + pinned `requirements.txt` + included data |
| **Best practice: hybrid search** | `src/retrieval.py` (`hybrid_search`) |
| **Best practice: re-ranking** | `src/retrieval.py` (`rerank`) |
| **Best practice: query rewriting** | `src/retrieval.py` (`rewrite_query`) |
| **Bonus: cloud deployment** | [Streamlit Cloud](#-deployment-streamlit-cloud) |

---

## 🙏 Data attribution

Tourism data © **Tourism Authority of Thailand (TAT)**. Sample data included for
educational/demo purposes.
