"""Preference Module: analyses the profile and retrieves matching catalog candidates."""
import pandas as pd
from sentence_transformers import util

from . import ui
from .config import CANDIDATE_POOL, CATALOG_PATH, SKILL_LEVELS
from .llm_client import chat, embedder

_catalog = None
_catalog_emb = None


def load_catalog():
    """Load the item catalog with Pandas and pre-compute item embeddings."""
    global _catalog, _catalog_emb
    if _catalog is None:
        _catalog = pd.read_csv(CATALOG_PATH)
        texts = (
            _catalog["title"] + ". " + _catalog["domain"] + ". "
            + _catalog["description"] + " Tags: " + _catalog["tags"]
        ).tolist()
        _catalog_emb = embedder().encode(texts, convert_to_tensor=True)
    return _catalog, _catalog_emb


def detect_format(preferences):
    text = (preferences or "").lower()
    if any(w in text for w in ("practical", "hands-on", "hands on", "project", "lab")):
        return "Practical"
    if any(w in text for w in ("theory", "theoretical", "concept")):
        return "Theoretical"
    return "Any"


def _matches_any(row, terms):
    haystack = f"{row['title']} {row['domain']} {row['tags']}".lower()
    return any(t.lower() in haystack for t in terms)


def retrieve_candidates(profile):
    """Semantic search over the catalog, filtered by category, level and dislikes."""
    catalog, catalog_emb = load_catalog()
    query = embedder().encode(profile.as_text(), convert_to_tensor=True)
    df = catalog.copy()
    df["similarity"] = util.cos_sim(query, catalog_emb)[0].cpu().numpy()

    # Small boosts for explicit level / format / liked-topic matches.
    level_idx = SKILL_LEVELS.index(profile.skill_level)
    df["level_gap"] = df["level"].map(SKILL_LEVELS.index) - level_idx
    df.loc[df["level_gap"] == 0, "similarity"] += 0.05
    df.loc[df["level_gap"] > 1, "similarity"] -= 0.15
    fmt = detect_format(profile.preferences)
    if fmt != "Any":
        df.loc[df["format"] == fmt, "similarity"] += 0.05
    if profile.liked:
        df.loc[df.apply(_matches_any, axis=1, terms=profile.liked), "similarity"] += 0.10

    if profile.disliked:
        df = df[~df.apply(_matches_any, axis=1, terms=profile.disliked)]
    if profile.preferred_category != "Any":
        preferred = df[df["type"] == profile.preferred_category]
        # Keep other types as backup when the preferred type has too few items.
        if len(preferred) >= profile.num_items:
            df = preferred
    return df.sort_values("similarity", ascending=False).head(CANDIDATE_POOL)


def analyze_preferences(profile):
    """Combine semantic domain affinity with an LLM reading of the profile."""
    candidates = retrieve_candidates(profile)
    domain_affinity = (
        candidates.groupby("domain")["similarity"].mean().sort_values(ascending=False)
    )
    prompt = (
        "Analyse this learner profile and respond in JSON with keys "
        '"key_interests" (list of strings), "learning_style" (string), '
        '"focus_areas" (list of 3 strings) and "summary" (one sentence).\n\n'
        f"Profile: {profile.as_text()}"
    )
    try:
        llm = chat(prompt, system="You are an expert academic counsellor.")
    except Exception as exc:  # LLM unavailable: fall back to rule-based analysis
        ui.warn(f"LLM analysis unavailable ({exc}); using rule-based analysis.")
        llm = {}
    return {
        "key_interests": llm.get("key_interests") or profile.interests,
        "learning_style": llm.get("learning_style") or detect_format(profile.preferences),
        "focus_areas": llm.get("focus_areas") or domain_affinity.index[:3].tolist(),
        "summary": llm.get("summary", ""),
        "preferred_format": detect_format(profile.preferences),
        "domain_affinity": {d: round(float(s), 3) for d, s in domain_affinity.items()},
        "candidates": candidates,
    }


def display_analysis(analysis):
    ui.section("PREFERENCE ANALYSIS")
    ui.kv("Key Interests", ", ".join(map(str, analysis["key_interests"])))
    ui.kv("Learning Style", analysis["learning_style"])
    ui.kv("Preferred Format", analysis["preferred_format"])
    ui.kv("Focus Areas", ", ".join(map(str, analysis["focus_areas"])))
    if analysis["summary"]:
        ui.kv("LLM Summary", analysis["summary"])
    print("\n  Domain affinity (Sentence-Transformer similarity):")
    for domain, score in analysis["domain_affinity"].items():
        bar = "#" * max(1, int(score * 40))
        print(f"    {domain:<26} {score:5.3f} {bar}")
    cands = analysis["candidates"]
    print(f"\n  Top {len(cands)} catalog candidates retrieved:")
    print(cands[["id", "title", "type", "level", "similarity"]]
          .round(3).to_string(index=False))
