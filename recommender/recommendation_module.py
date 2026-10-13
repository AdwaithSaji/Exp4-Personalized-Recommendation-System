"""Recommendation Module: sends the prompt to the LLM and validates its answer."""
import pandas as pd

from . import ui
from .config import LLM_MODEL
from .llm_client import chat
from .prompt_module import SYSTEM_PROMPT


def generate_recommendations(prompt, analysis, num_items):
    candidates = analysis["candidates"].set_index("id")
    picked = []
    try:
        reply = chat(prompt, system=SYSTEM_PROMPT)
        for item in reply.get("recommendations", []):
            try:
                item_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            if item_id in candidates.index and item_id not in [p["id"] for p in picked]:
                picked.append({
                    "id": item_id,
                    "llm_reason": str(item.get("reason", "")).strip(),
                    "llm_confidence": float(item.get("confidence", 70) or 70),
                })
    except Exception as exc:
        ui.warn(f"LLM call failed ({exc}); falling back to semantic candidates.")

    # Top up from the semantic candidates if the LLM returned too few valid items.
    for item_id in candidates.index:
        if len(picked) >= num_items:
            break
        if item_id not in [p["id"] for p in picked]:
            picked.append({"id": int(item_id),
                           "llm_reason": "Strong semantic match with the user profile.",
                           "llm_confidence": 60.0})

    recs = pd.DataFrame(picked[:num_items])
    return recs.merge(candidates.reset_index(), on="id", how="left")


def display_recommendations(recs, title="GPT-4 RECOMMENDATION"):
    ui.section(f"{title}  (model: {LLM_MODEL} via Ollama)")
    for i, r in enumerate(recs.itertuples(), 1):
        print(f"  {i}. {r.title}  [{r.type} | {r.level} | {r.format}]")
        ui.wrap(f"LLM reason: {r.llm_reason}  (confidence {r.llm_confidence:.0f}%)", indent=6)
