"""Save Module: stores recommendations and session history with Pandas."""
import json
from datetime import datetime

from . import ui
from .config import OUTPUT_DIR

COLUMNS = ["rank", "title", "type", "domain", "level", "format", "duration_hours",
           "score", "llm_confidence", "semantic", "matched_interests", "explanation"]


def save_recommendations(profile, ranked, feedback_history):
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUTPUT_DIR / f"recommendations_{profile.name.replace(' ', '_')}_{stamp}"

    cols = [c for c in COLUMNS if c in ranked.columns]
    ranked[cols].to_csv(base.with_suffix(".csv"), index=False)
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump({
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "profile": profile.to_dict(),
            "feedback_history": feedback_history,
            "recommendations": ranked[cols].to_dict(orient="records"),
        }, fh, indent=2)

    ui.section("SAVE RECOMMENDATIONS")
    ui.success("Recommendation saved successfully!")
    ui.kv("CSV file", base.with_suffix(".csv").relative_to(OUTPUT_DIR.parent))
    ui.kv("JSON file", base.with_suffix(".json").relative_to(OUTPUT_DIR.parent))
    ui.kv("Items saved", len(ranked))
    return base
