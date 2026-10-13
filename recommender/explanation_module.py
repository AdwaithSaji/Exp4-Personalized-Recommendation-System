"""Explanation Module: explains why each item was recommended."""
from . import ui
from .llm_client import chat


def matched_interests(row, profile):
    text = f"{row['title']} {row['tags']} {row['domain']} {row['description']}".lower()
    hits = []
    for interest in profile.interests + profile.liked:
        words = [w for w in interest.lower().split() if len(w) > 1]
        if interest.lower() in text or (words and all(w in text for w in words)):
            hits.append(interest)
    return list(dict.fromkeys(hits))


def explain_recommendations(ranked, profile):
    items = "\n".join(f"{r.rank}. {r.title} ({r.level}, {r.format}): {r.description}"
                      for r in ranked.itertuples())
    prompt = (
        f"User profile: {profile.as_text()}\n\nRecommended items:\n{items}\n\n"
        "For every item write a two-sentence personalised explanation of why it suits "
        "this user and what they will gain. Respond in JSON as "
        '{"explanations": [{"rank": <rank>, "explanation": "<text>"}]}'
    )
    by_rank = {}
    try:
        reply = chat(prompt, system="You are a friendly academic mentor.")
        for e in reply.get("explanations", []):
            try:
                by_rank[int(e["rank"])] = str(e["explanation"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
    except Exception as exc:
        ui.warn(f"LLM explanation failed ({exc}); using LLM selection reasons.")

    explained = ranked.copy()
    explained["explanation"] = [by_rank.get(r.rank) or r.llm_reason
                                for r in ranked.itertuples()]
    explained["matched_interests"] = [", ".join(matched_interests(row, profile)) or "-"
                                      for _, row in ranked.iterrows()]
    return explained


def display_explanations(explained):
    ui.section("RECOMMENDATION EXPLANATION")
    for r in explained.itertuples():
        print(f"  #{r.rank} {r.title}  -  Score {r.score:.1f}%")
        ui.kv("Matched interests", r.matched_interests, width=18)
        ui.kv("Level / Format", f"{r.level} / {r.format} ({r.duration_hours} hrs)", width=18)
        ui.kv("Why", r.explanation, width=18)
        print()
