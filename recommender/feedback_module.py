"""Feedback Module: interprets user feedback and updates the profile for refinement."""
import re

from . import ui
from .config import SKILL_LEVELS
from .llm_client import chat


def collect_feedback():
    ui.section("USER FEEDBACK")
    return input("  Your feedback on these recommendations: ").strip()


LEVEL_UP = ("too easy", "too basic", "more advanced", "harder", "challenging")
LEVEL_DOWN = ("too hard", "too difficult", "too advanced", "easier", "simpler")
DIFFICULTY_WORDS = ("easy", "easier", "hard", "difficult", "advanced", "basic",
                    "challenging", "simple")


def _rule_based(feedback):
    """Fallback parser for phrases like 'more interested in X than Y'."""
    text = feedback.lower()
    parsed = {"increase": [], "decrease": [], "skill_level": None}
    m = (re.search(r"more (?:interested in|into) (.+?) than (.+?)[.!]?$", text)
         or re.search(r"prefer (.+?) (?:over|to) (.+?)[.!]?$", text))
    if m:
        parsed["increase"].append(m.group(1).strip())
        parsed["decrease"].append(m.group(2).strip())
    for m in re.finditer(r"(?:not interested in|don't want|no more|avoid) (.+?)(?:[.,!]|$)", text):
        parsed["decrease"].append(m.group(1).strip())
    if any(k in text for k in LEVEL_UP):
        parsed["skill_level"] = "up"
    elif any(k in text for k in LEVEL_DOWN):
        parsed["skill_level"] = "down"
    return parsed


def interpret_feedback(feedback):
    prompt = (
        f'A learner gave this feedback on their recommendations: "{feedback}"\n'
        "Extract the topics they want more of and the topics they want less of. "
        "Use short subject names exactly as the learner wrote them; difficulty words "
        "such as easy or advanced are not topics. Respond in JSON as "
        '{"increase": [topics], "decrease": [topics], '
        '"summary": "<one sentence>"}'
    )
    try:
        parsed = chat(prompt, system="You extract structured preferences from feedback.")
    except Exception as exc:
        ui.warn(f"LLM feedback parsing failed ({exc}); using rule-based parser.")
        parsed = {}
    rules = _rule_based(feedback)

    def clean(xs):
        # Difficulty words are handled as a level change, not as topics.
        return [str(x).strip() for x in (xs or []) if str(x).strip()
                and not any(w in str(x).lower() for w in DIFFICULTY_WORDS)]
    return {
        "increase": clean(parsed.get("increase")) or rules["increase"],
        "decrease": clean(parsed.get("decrease")) or rules["decrease"],
        # Level changes are keyword-driven so the LLM cannot invent one.
        "skill_level": rules["skill_level"],
        "summary": parsed.get("summary", ""),
    }


def apply_feedback(profile, parsed):
    """Update the profile in place so the next retrieval reflects the feedback."""
    for topic in parsed["increase"]:
        if topic.lower() not in [t.lower() for t in profile.liked]:
            profile.liked.append(topic)
        profile.disliked = [t for t in profile.disliked if t.lower() != topic.lower()]
    for topic in parsed["decrease"]:
        if topic.lower() not in [t.lower() for t in profile.disliked]:
            profile.disliked.append(topic)
        profile.liked = [t for t in profile.liked if t.lower() != topic.lower()]
        profile.interests = [i for i in profile.interests if i.lower() != topic.lower()]
    idx = SKILL_LEVELS.index(profile.skill_level)
    if parsed["skill_level"] == "up" and idx < len(SKILL_LEVELS) - 1:
        profile.skill_level = SKILL_LEVELS[idx + 1]
    elif parsed["skill_level"] == "down" and idx > 0:
        profile.skill_level = SKILL_LEVELS[idx - 1]
    return profile


def display_feedback(feedback, parsed, profile):
    ui.kv("Feedback", f'"{feedback}"')
    ui.kv("Increase", ", ".join(parsed["increase"]) or "-")
    ui.kv("Decrease", ", ".join(parsed["decrease"]) or "-")
    ui.kv("Skill level change", parsed["skill_level"] or "none")
    if parsed["summary"]:
        ui.kv("Interpretation", parsed["summary"])
    ui.success(f"Profile updated -> boosted: {', '.join(profile.liked) or '-'} | "
               f"avoided: {', '.join(profile.disliked) or '-'} | level: {profile.skill_level}")


def display_comparison(before, after):
    ui.section("REFINED RECOMMENDATIONS")
    old, new = set(before["title"]), set(after["title"])
    print(f"  {'Rank':<5}{'Recommendation':<44}{'Score':>8}   Change")
    print("  " + "-" * 70)
    for r in after.itertuples():
        tag = "kept" if r.title in old else "NEW"
        print(f"  {r.rank:<5}{r.title[:42]:<44}{r.score:>7.1f}%   {tag}")
    removed = old - new
    if removed:
        print(f"\n  Removed after feedback: {', '.join(sorted(removed))}")
