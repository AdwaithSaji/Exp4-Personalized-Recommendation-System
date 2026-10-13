"""Intent Module: classifies free-text user messages with sentence embeddings."""
from sentence_transformers import util

from . import ui
from .llm_client import embedder

INTENT_EXAMPLES = {
    "get_recommendations": [
        "recommend me some courses", "what should I learn next",
        "suggest something for me", "give me recommendations",
    ],
    "refine_recommendations": [
        "I am more interested in AI than web development",
        "I don't like these suggestions", "show me more items like the first one",
        "remove web development from my list",
    ],
    "explain_recommendation": [
        "why did you recommend this", "explain the first recommendation",
        "how is this course relevant to me", "what is the reason behind this suggestion",
        "why is this item ranked first", "why was Python for AI suggested",
    ],
    "change_skill_level": [
        "these are too easy for me", "these courses are too basic",
        "give me more advanced material", "I want harder and more challenging courses",
        "this is too difficult for me", "I want beginner friendly content",
        "the level is not right for me",
    ],
    "save_recommendations": [
        "save my recommendations", "export this list to a file",
        "store these results", "download the recommendations",
    ],
    "view_profile": ["show my profile", "what details do you have about me",
                     "display my preferences"],
    "exit": ["bye", "quit the application", "exit", "I am done, thank you"],
}

_intent_emb = None


def _examples():
    global _intent_emb
    if _intent_emb is None:
        labels, texts = [], []
        for intent, examples in INTENT_EXAMPLES.items():
            labels += [intent] * len(examples)
            texts += examples
        _intent_emb = (labels, embedder().encode(texts, convert_to_tensor=True))
    return _intent_emb


def detect_intent(message):
    """Return (best_intent, confidence, ranked list of (intent, score))."""
    labels, example_emb = _examples()
    query = embedder().encode(message, convert_to_tensor=True)
    sims = util.cos_sim(query, example_emb)[0].cpu().numpy()
    best = {}
    for label, sim in zip(labels, sims):
        best[label] = max(best.get(label, -1.0), float(sim))
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    intent, score = ranked[0]
    if score < 0.35:
        intent = "unknown"
    return intent, score, ranked


def display_intent(message, intent, score, ranked):
    ui.section("INTENT DETECTION")
    ui.kv("User message", f'"{message}"')
    ui.kv("Detected intent", intent.upper())
    ui.kv("Confidence", f"{score * 100:.1f}%")
    print("\n  Similarity to each intent:")
    for label, s in ranked:
        print(f"    {label:<24} {s:5.3f} {'#' * max(1, int(s * 40))}")
