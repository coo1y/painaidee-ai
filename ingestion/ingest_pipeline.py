"""Automated ingestion pipeline (Prefect flow).

Reads the raw TAT exports, cleans HTML, normalizes fields, and loads two
ChromaDB collections (``attractions`` and ``events``) with searchable metadata
(province, category, dates as epoch-days, coordinates, contacts).

Run as an orchestrated flow:
    python -m ingestion.ingest_pipeline

Or from Python:
    from ingestion.ingest_pipeline import ingest_flow
    ingest_flow(reset=True)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
# Allow running as a plain script (python ingestion/ingest_pipeline.py).
sys.path.insert(0, str(_ROOT))

# Isolate Prefect's local state inside the project so it never collides with a
# different Prefect version's ~/.prefect database (reproducible on any machine).
os.environ.setdefault("PREFECT_HOME", str(_ROOT / ".prefect"))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from src import config
from src.embeddings import backend_name, embed_texts
from src.utils import (
    clean_html,
    clean_listagg,
    clean_text,
    first_nonempty,
    load_tat_export,
    parse_date,
    parse_latlng,
    to_epoch_day,
)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def _sanitize_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata must be str/int/float/bool and non-null."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None or v == "":
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def normalize_attraction(rec: dict[str, Any]) -> dict[str, Any]:
    name_th = clean_text(rec.get("ATT_NAME_TH"))
    name_en = clean_text(rec.get("ATT_NAME_EN"))
    detail_th = clean_html(rec.get("ATT_DETAIL_TH"))
    detail_en = clean_html(rec.get("ATT_DETAIL_EN"))
    highlight = clean_html(rec.get("ATT_HILIGHT"))
    province = clean_text(rec.get("PROVINCE_NAME_TH"))
    district = clean_text(rec.get("DISTRICT_NAME_TH"))
    region = clean_text(rec.get("REGION_NAME_TH"))
    category = clean_text(rec.get("ATT_CATEGORY_LABEL"))
    subtype = clean_text(rec.get("ATT_TYPE_LABEL"))
    hours = clean_text(rec.get("ATT_START_END"))
    lat, lng = parse_latlng(rec.get("ATT_LOCATION"))

    # Document text embeds Thai originals plus English where present, so the
    # multilingual embedding model can match both Thai and English queries.
    document = "\n".join(
        p
        for p in [
            f"ชื่อ/Name: {name_th} ({name_en})",
            f"จังหวัด/Province: {province} | อำเภอ/District: {district} | ภาค/Region: {region}",
            f"ประเภท/Category: {category} - {subtype}",
            f"จุดเด่น/Highlight: {highlight}" if highlight else "",
            f"รายละเอียด/Details (TH): {detail_th}" if detail_th else "",
            f"Details (EN): {detail_en}" if detail_en else "",
            f"เวลาเปิด/Opening: {hours}" if hours else "",
        ]
        if p
    )

    meta = _sanitize_meta(
        {
            "source_type": "attraction",
            "att_id": clean_text(rec.get("ATT_ID")),
            "name_en": name_en,
            "name_th": name_th,
            "province": province,
            "district": district,
            "region": region,
            "category": category,
            "subtype": subtype,
            "opening_hours": hours,
            "latitude": lat,
            "longitude": lng,
            "tel": clean_text(rec.get("ATT_TEL")),
            "website": clean_text(rec.get("ATT_WEBSITE")),
            "facebook": clean_text(rec.get("ATT_FACEBOOK")),
            "highlight": highlight[:500],
        }
    )
    return {"id": f"att-{rec.get('ATT_ID')}", "document": document, "metadata": meta}


def normalize_event(rec: dict[str, Any]) -> dict[str, Any]:
    name = first_nonempty(rec.get("NAME"), rec.get("NAME_EN"))
    name_en = clean_text(rec.get("NAME_EN"))
    desc_th = clean_html(rec.get("DESCRIPTION"))
    desc_en = clean_html(rec.get("DESCRIPTION_EN"))
    province = clean_text(rec.get("PROVINCE"))
    district = clean_text(rec.get("DISTRICT"))
    region = clean_text(rec.get("REGION"))
    etype = clean_text(rec.get("TATEVENTTYPENAME"))
    venue = clean_text(rec.get("TATEVENTPLACEDETAIL"))
    start_iso = parse_date(rec.get("STARTDATE"))
    end_iso = parse_date(rec.get("ENDDATE"))
    target = clean_listagg(rec.get("EVENTTARGETGROUP"))
    interested = clean_listagg(rec.get("EVENTINTERESTED"))
    lat, lng = parse_latlng(rec.get("LOCATION"))
    price_th = clean_text(rec.get("TICKETPRICEADULTTH"))

    document = "\n".join(
        p
        for p in [
            f"งาน/Event: {name}",
            f"ประเภท/Type: {etype}" if etype else "",
            f"จังหวัด/Province: {province} | อำเภอ/District: {district} | ภาค/Region: {region}",
            f"สถานที่/Venue: {venue}" if venue else "",
            f"ช่วงวันที่/Dates: {start_iso} to {end_iso}" if start_iso else "",
            f"รายละเอียด/Details (TH): {desc_th}" if desc_th else "",
            f"Details (EN): {desc_en}" if desc_en else "",
            f"กลุ่มเป้าหมาย/Target: {', '.join(target)}" if target else "",
            f"ความสนใจ/Interested: {', '.join(interested)}" if interested else "",
        ]
        if p
    )

    meta = _sanitize_meta(
        {
            "source_type": "event",
            "event_id": clean_text(rec.get("SYSTEMPREVIOUSVERSIONID")),
            "name": name,
            "name_en": name_en,
            "event_type": etype,
            "province": province,
            "district": district,
            "region": region,
            "venue": venue,
            "start_date": start_iso,
            "end_date": end_iso,
            "start_day": to_epoch_day(start_iso),
            "end_day": to_epoch_day(end_iso),
            "target_group": ", ".join(target),
            "interested": ", ".join(interested),
            "ticket_price_th": price_th,
            "latitude": lat,
            "longitude": lng,
            "phone": clean_text(rec.get("PHONE")),
            "website": clean_text(rec.get("WEBSITE")),
            "facebook": clean_text(rec.get("FACEBOOK")),
        }
    )
    return {
        "id": f"evt-{rec.get('SYSTEMPREVIOUSVERSIONID')}",
        "document": document,
        "metadata": meta,
    }


# --------------------------------------------------------------------------
# Core logic (plain functions — no Prefect, importable by the app bootstrap)
# --------------------------------------------------------------------------
def load_and_normalize_core(path: str, kind: str) -> list[dict[str, Any]]:
    records = load_tat_export(path)
    fn = normalize_attraction if kind == "attraction" else normalize_event
    normalized = [fn(r) for r in records]
    # Drop records without any usable text.
    return [n for n in normalized if n["document"].strip()]


def build_collection_core(
    chroma_path: str, name: str, items: list[dict[str, Any]], reset: bool
) -> int:
    """Build/refresh one Chroma collection. Creates its own client (picklable args)."""
    client = chromadb.PersistentClient(
        path=chroma_path, settings=Settings(anonymized_telemetry=False)
    )
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )
    if not items:
        return 0
    documents = [it["document"] for it in items]
    embeddings = embed_texts(documents)
    collection.upsert(
        ids=[it["id"] for it in items],
        documents=documents,
        metadatas=[it["metadata"] for it in items],
        embeddings=embeddings,
    )
    return collection.count()


def run_ingestion(reset: bool = True) -> dict[str, int]:
    """Plain ingestion entrypoint (used by the Streamlit app's cold-start bootstrap)."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_path = str(config.CHROMA_DIR)
    attractions = load_and_normalize_core(str(config.ATTRACTION_FILE), "attraction")
    events = load_and_normalize_core(str(config.ACTIVITY_FILE), "event")
    return {
        config.ATTRACTIONS_COLLECTION: build_collection_core(
            chroma_path, config.ATTRACTIONS_COLLECTION, attractions, reset
        ),
        config.EVENTS_COLLECTION: build_collection_core(
            chroma_path, config.EVENTS_COLLECTION, events, reset
        ),
    }


# --------------------------------------------------------------------------
# Prefect wrappers (orchestrated pipeline for CLI / docker)
# --------------------------------------------------------------------------
@task(retries=1, cache_policy=NO_CACHE)
def load_and_normalize(path: str, kind: str) -> list[dict[str, Any]]:
    return load_and_normalize_core(path, kind)


@task(cache_policy=NO_CACHE)
def build_collection(chroma_path: str, name: str, items: list[dict[str, Any]], reset: bool) -> int:
    n = build_collection_core(chroma_path, name, items, reset)
    get_run_logger().info("Collection '%s' now has %d documents", name, n)
    return n


@flow(name="painaidee-ingestion")
def ingest_flow(reset: bool = True) -> dict[str, int]:
    logger = get_run_logger()
    logger.info("Embedding backend: %s", backend_name())
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_path = str(config.CHROMA_DIR)

    attractions = load_and_normalize(str(config.ATTRACTION_FILE), "attraction")
    events = load_and_normalize(str(config.ACTIVITY_FILE), "event")

    n_att = build_collection(chroma_path, config.ATTRACTIONS_COLLECTION, attractions, reset)
    n_evt = build_collection(chroma_path, config.EVENTS_COLLECTION, events, reset)

    result = {"attractions": n_att, "events": n_evt}
    logger.info("Ingestion complete: %s", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest TAT data into ChromaDB")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not delete existing collections before ingesting.",
    )
    args = parser.parse_args()
    result = ingest_flow(reset=not args.no_reset)
    print(f"Done. Ingested: {result}")


if __name__ == "__main__":
    main()
