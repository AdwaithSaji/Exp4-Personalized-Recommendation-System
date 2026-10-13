"""Input Module: reads and validates user details from the console."""
from . import ui
from .config import CATEGORIES, SKILL_LEVELS


def ask(prompt, default=None, required=True):
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"  {prompt}{suffix}: ").strip()
        if not value and default is not None:
            return default
        if value or not required:
            return value
        ui.warn("This field is required.")


def ask_int(prompt, default, low, high):
    while True:
        raw = ask(prompt, str(default))
        if raw.isdigit() and low <= int(raw) <= high:
            return int(raw)
        ui.warn(f"Enter a number between {low} and {high}.")


def ask_choice(prompt, choices, default):
    options = "/".join(choices)
    while True:
        raw = ask(f"{prompt} ({options})", default)
        for choice in choices:
            if raw.lower() == choice.lower():
                return choice
        ui.warn(f"Choose one of: {options}")


def collect_user_input():
    ui.section("USER PROFILE INPUT")
    return {
        "name": ask("Name"),
        "age": ask_int("Age", 20, 10, 100),
        "background": ask("Background (e.g. Computer Science student)"),
        "interests": ask("Interests (comma separated)"),
        "skill_level": ask_choice("Skill Level", SKILL_LEVELS, "Beginner"),
        "preferred_category": ask_choice("Preferred Category", CATEGORIES, "Any"),
        "goal": ask("Goal"),
        "preferences": ask("Preferences (e.g. practical, short courses)", "", required=False),
        "num_items": ask_int("Number of items to recommend", 5, 1, 10),
    }
