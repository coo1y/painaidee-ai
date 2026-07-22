"""LLM output evaluation: compare two prompt variants with an LLM judge.

Two system prompts are compared on a set of questions:
  - A: "concise" assistant
  - B: "detailed planner" assistant
Both receive the SAME retrieved context. An LLM judge scores each answer 1-5 on
faithfulness (grounded in context) and helpfulness. The higher-average prompt
wins. Requires OPENAI_API_KEY.

Run:  python -m eval.llm_eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.agent import _format_kb_source
from src.llm import chat, chat_json
from src.retrieval import retrieve

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"

DEFAULT_QUESTIONS = [
    "Recommend a cultural riverside old town to visit.",
    "Where can I experience local community life and traditional crafts?",
    "Are there any festivals or events around New Year in eastern Thailand?",
    "Suggest a historic destination in northern Thailand.",
]

PROMPT_A = (
    "You are PaiNaiDee, a Thailand travel assistant. Answer in English, concisely "
    "(2-4 sentences). Use ONLY the provided context and cite sources as [S#]. "
    "Do not invent details."
)
PROMPT_B = (
    "You are PaiNaiDee, an expert Thailand trip planner. Answer in English with a "
    "short intro then bullet points (name, why to go, practical tip). Use ONLY the "
    "provided context and cite sources as [S#]. If a detail is missing, say so."
)

JUDGE_SYS = (
    "You are a strict evaluator. Given a question, the retrieved CONTEXT, and an "
    "ANSWER, rate the answer. Respond ONLY as JSON: "
    "{\"faithfulness\": 1-5, \"helpfulness\": 1-5}. Faithfulness = grounded in the "
    "context with correct citations; helpfulness = useful and clear for a traveler."
)


def load_questions() -> list[str]:
    if QUESTIONS_PATH.exists():
        return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return DEFAULT_QUESTIONS


def build_context(question: str) -> str:
    hits = retrieve(question)["results"]
    return "\n\n".join(_format_kb_source(i + 1, h) for i, h in enumerate(hits)) or "(none)"


def generate(system_prompt: str, question: str, context: str) -> str:
    return chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
        temperature=0.3,
    )


def judge(question: str, context: str, answer: str) -> dict:
    out = chat_json(
        [
            {"role": "system", "content": JUDGE_SYS},
            {
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}",
            },
        ]
    )
    return {
        "faithfulness": float(out.get("faithfulness", 0) or 0),
        "helpfulness": float(out.get("helpfulness", 0) or 0),
    }


def main() -> None:
    if not config.has_openai():
        print("OPENAI_API_KEY not set — LLM evaluation requires OpenAI. Skipping.")
        return

    questions = load_questions()
    variants = {"A (concise)": PROMPT_A, "B (detailed)": PROMPT_B}
    totals = {name: {"faithfulness": 0.0, "helpfulness": 0.0} for name in variants}

    for q in questions:
        context = build_context(q)
        for name, prompt in variants.items():
            ans = generate(prompt, q, context)
            scores = judge(q, context, ans)
            totals[name]["faithfulness"] += scores["faithfulness"]
            totals[name]["helpfulness"] += scores["helpfulness"]

    n = max(len(questions), 1)
    print(f"Questions evaluated: {len(questions)}\n")
    print(f"{'variant':<16}{'faithfulness':>14}{'helpfulness':>13}{'overall':>10}")
    print("-" * 53)
    averages = {}
    for name, t in totals.items():
        f, h = t["faithfulness"] / n, t["helpfulness"] / n
        overall = (f + h) / 2
        averages[name] = overall
        print(f"{name:<16}{f:>14.2f}{h:>13.2f}{overall:>10.2f}")

    best = max(averages, key=averages.get)
    print(f"\nBest prompt: {best}")


if __name__ == "__main__":
    main()
