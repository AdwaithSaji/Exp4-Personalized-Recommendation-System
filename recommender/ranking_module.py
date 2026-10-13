"""Ranking Module: scores recommendations by combining LLM and semantic signals."""
from sentence_transformers import util

from . import ui
from .config import SKILL_LEVELS
from .llm_client import embedder
from .preference_module import detect_format

# Weights of the final suitability score.
W_LLM, W_SEMANTIC, W_LEVEL, W_FORMAT = 0.40, 0.35, 0.15, 0.10


def rank_recommendations(recs, profile):
    model = embedder()
    profile_emb = model.encode(profile.as_text(), convert_to_tensor=True)
    item_emb = model.encode(
        (recs["title"] + ". " + recs["description"]).tolist(), convert_to_tensor=True)
    sims = util.cos_sim(profile_emb, item_emb)[0].cpu().numpy()

    ranked = recs.copy()
    # MiniLM cosine scores rarely exceed ~0.7, so stretch them onto 0-100.
    ranked["semantic"] = [min(100.0, max(0.0, float(s) * 140)) for s in sims]
    user_level = SKILL_LEVELS.index(profile.skill_level)
    ranked["level_fit"] = ranked["level"].map(
        lambda lvl: 100 - 50 * abs(SKILL_LEVELS.index(lvl) - user_level))
    fmt = detect_format(profile.preferences)
    ranked["format_fit"] = ranked["format"].map(
        lambda f: 100 if fmt == "Any" or f == fmt else (60 if f == "Mixed" else 30))
    ranked["score"] = (
        W_LLM * ranked["llm_confidence"] + W_SEMANTIC * ranked["semantic"]
        + W_LEVEL * ranked["level_fit"] + W_FORMAT * ranked["format_fit"]
    ).round(1)
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def display_ranking(ranked, profile, title="RECOMMENDATION RANKING"):
    ui.section(title)
    print(f"  User: {profile.background}")
    print(f"  Interests: {', '.join(profile.interests)}")
    print(f"  Level: {profile.skill_level}\n")
    header = (f"  {'Rank':<5}{'Recommendation':<42}{'Type':<14}"
              f"{'LLM':>5}{'Sem':>6}{'Lvl':>6}{'Score':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in ranked.itertuples():
        print(f"  {r.rank:<5}{r.title[:40]:<42}{r.type:<14}{r.llm_confidence:>5.0f}"
              f"{r.semantic:>6.0f}{r.level_fit:>6.0f}{r.score:>7.1f}%")
    print(f"\n  Score = {W_LLM:.2f} x LLM confidence + {W_SEMANTIC:.2f} x semantic similarity"
          f" + {W_LEVEL:.2f} x level fit + {W_FORMAT:.2f} x format fit")
