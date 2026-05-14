"""
LLM-backed answer generation and LLM-as-judge scoring via Gemini.

Both call through the cache layer so a re-run with the same prompt or the
same (query, predicted, gold, judge_model) triple is free.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional

from cache import AnswerCache, JudgeCache


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY missing")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


_ANSWER_CACHE = AnswerCache()
_JUDGE_CACHE = JudgeCache()


def _call_with_retries(model: str, prompt: str, max_retries: int = 5) -> str:
    client = _client()
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return (resp.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            transient = any(k in msg for k in ("429", "503", "500", "504", "resource_exhausted", "deadline"))
            if attempt == max_retries - 1 or not transient:
                raise
            wait = (2 ** attempt) * 1.5
            print(f"  [llm] transient error '{exc}' — retry in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def build_mrag_prompt(query: str, primary_chunks: List[str], structural_chunks: List[str]) -> str:
    """Paper §6 prompt template."""
    primary = "\n".join(f"- {c}" for c in primary_chunks) or "(none)"
    structural = "\n".join(f"- {c}" for c in structural_chunks) or "(none)"
    return f"""<primary_semantic_context>
{primary}
</primary_semantic_context>

<structurally_linked_context>
{structural}
</structurally_linked_context>

System: Synthesize the primary context to answer the user's query.
Use the structurally linked context to draw broader multi-hop connections if the primary context is missing direct evidence.
Answer the question with a short factual span only — no preamble, no explanation, no period. If you cannot answer, output "unknown".

Question: {query}
Short answer:"""


def build_flat_prompt(query: str, chunks: List[str]) -> str:
    """Baseline naive-RAG prompt — no hop structure."""
    ctx = "\n".join(f"- {c}" for c in chunks) or "(none)"
    return f"""<context>
{ctx}
</context>

Answer the question with a short factual span only — no preamble, no explanation, no period. If you cannot answer, output "unknown".

Question: {query}
Short answer:"""


def build_no_context_prompt(query: str) -> str:
    """Sanity baseline — no retrieval at all."""
    return f"""Answer the question with a short factual span only — no preamble, no explanation, no period. If you cannot answer, output "unknown".

Question: {query}
Short answer:"""


def generate_answer(prompt: str, model: str = "gemini-2.5-flash") -> str:
    key = AnswerCache.key(prompt, model)
    hit = _ANSWER_CACHE.get(key)
    if hit is not None:
        return hit["answer"]
    text = _call_with_retries(model, prompt)
    # Strip trailing period the model sometimes adds despite the instruction.
    answer = text.rstrip(".").strip().strip('"').strip("'")
    _ANSWER_CACHE.put(key, {"answer": answer, "model": model})
    return answer


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are a strict evaluator for short-answer QA.

The gold answer below is from a multi-hop QA benchmark. Mark the predicted
answer CORRECT if it conveys the same factual meaning as the gold answer —
accept differences in casing, articles, punctuation, abbreviation,
paraphrasing, or partial spans that uniquely identify the gold entity.
Mark INCORRECT if the prediction contradicts the gold, introduces a different
entity, hedges with "unknown" when the gold is a real entity, or is too
ambiguous to identify the gold answer.

Question: {query}
Gold answer: {gold}
Predicted answer: {pred}

Reply with ONLY a JSON object like {{"correct": true_or_false, "rationale": "one short sentence"}}."""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def llm_judge(query: str, predicted: str, gold: str, model: str = "gemini-2.5-flash") -> dict:
    key = JudgeCache.key(query, predicted, gold, model)
    hit = _JUDGE_CACHE.get(key)
    if hit is not None:
        return hit

    prompt = _JUDGE_PROMPT.format(query=query, gold=gold, pred=predicted)
    raw = _call_with_retries(model, prompt)
    parsed = {"correct": False, "rationale": "judge parse failed", "raw": raw[:500]}
    m = _JSON_BLOCK.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            parsed = {"correct": bool(obj.get("correct", False)),
                      "rationale": str(obj.get("rationale", "")),
                      "model": model}
        except json.JSONDecodeError:
            pass
    _JUDGE_CACHE.put(key, parsed)
    return parsed
