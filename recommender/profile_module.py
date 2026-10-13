"""Profile Module: builds and displays a structured user profile."""
from dataclasses import asdict, dataclass, field

from . import ui


@dataclass
class UserProfile:
    name: str
    age: int
    background: str
    interests: list
    skill_level: str
    preferred_category: str
    goal: str
    preferences: str
    num_items: int
    liked: list = field(default_factory=list)
    disliked: list = field(default_factory=list)

    def as_text(self):
        """Natural-language summary used for prompts and embeddings."""
        text = (
            f"{self.background} at {self.skill_level} level interested in "
            f"{', '.join(self.interests)}. Goal: {self.goal}."
        )
        if self.preferences:
            text += f" Preferences: {self.preferences}."
        if self.liked:
            text += f" More interested in {', '.join(self.liked)}."
        return text

    def to_dict(self):
        return asdict(self)


def build_profile(raw):
    interests = [i.strip() for i in raw["interests"].split(",") if i.strip()]
    return UserProfile(
        name=raw["name"].title(),
        age=raw["age"],
        background=raw["background"],
        interests=interests,
        skill_level=raw["skill_level"],
        preferred_category=raw["preferred_category"],
        goal=raw["goal"],
        preferences=raw["preferences"],
        num_items=raw["num_items"],
    )


def display_profile(profile):
    ui.section("COLLECTED USER PROFILE")
    ui.kv("Name", profile.name)
    ui.kv("Age", profile.age)
    ui.kv("Background", profile.background)
    ui.kv("Interests", ", ".join(profile.interests))
    ui.kv("Skill Level", profile.skill_level)
    ui.kv("Preferred Category", profile.preferred_category)
    ui.kv("Goal", profile.goal)
    ui.kv("Preferences", profile.preferences or "-")
    ui.kv("Number of items", profile.num_items)
    if profile.liked:
        ui.kv("Boosted topics", ", ".join(profile.liked))
    if profile.disliked:
        ui.kv("Avoided topics", ", ".join(profile.disliked))
