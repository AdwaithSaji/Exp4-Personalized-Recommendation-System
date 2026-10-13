"""Prompt Module: converts the profile and analysis into a structured LLM prompt."""
from . import ui

SYSTEM_PROMPT = (
    "You are a personalized learning recommendation engine. You only recommend "
    "items from the candidate list you are given and you always reply in valid JSON."
)


def build_prompt(profile, analysis, feedback=None, previous=None):
    candidates = "\n".join(
        f"- id={r.id} | {r.title} | {r.type} | {r.level} | {r.format} | {r.domain}"
        for r in analysis["candidates"].itertuples()
    )
    lines = [
        "USER PROFILE",
        f"Name: {profile.name}",
        f"Age: {profile.age}",
        f"Background: {profile.background}",
        f"Interests: {', '.join(profile.interests)}",
        f"Skill Level: {profile.skill_level}",
        f"Preferred Category: {profile.preferred_category}",
        f"Goal: {profile.goal}",
        f"Preferences: {profile.preferences or 'None'}",
        "",
        "PREFERENCE ANALYSIS",
        f"Key interests: {', '.join(map(str, analysis['key_interests']))}",
        f"Focus areas: {', '.join(map(str, analysis['focus_areas']))}",
        f"Learning style: {analysis['learning_style']}",
    ]
    if profile.liked:
        lines.append(f"Prioritise topics: {', '.join(profile.liked)}")
    if profile.disliked:
        lines.append(f"Avoid topics: {', '.join(profile.disliked)}")
    if feedback:
        lines += ["", "USER FEEDBACK ON PREVIOUS RECOMMENDATIONS",
                  f"Previous list: {', '.join(previous or [])}",
                  f'Feedback: "{feedback}"',
                  "Refine the list so that it reflects this feedback."]
    lines += [
        "",
        "CANDIDATE ITEMS",
        candidates,
        "",
        "TASK",
        f"Select the {profile.num_items} most suitable candidate items for this user, "
        "best first. Respond with JSON of the form:",
        '{"recommendations": [{"id": <candidate id>, "title": "<title>", '
        '"reason": "<one short sentence>", "confidence": <0-100>}]}',
    ]
    return "\n".join(lines)


def display_prompt(prompt):
    ui.section("PROMPT GENERATION")
    print("  [system]")
    ui.wrap(SYSTEM_PROMPT, indent=4)
    print("  [user]")
    for line in prompt.splitlines():
        print(f"    {line}")
