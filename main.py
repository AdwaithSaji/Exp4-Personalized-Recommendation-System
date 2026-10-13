"""Main Module: menu-driven Personalized Recommendation System (Experiment 4)."""
import os
import sys
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

from recommender import ui
from recommender.config import EMBEDDING_MODEL, LLM_MODEL
from recommender.explanation_module import display_explanations, explain_recommendations
from recommender.feedback_module import (apply_feedback, collect_feedback, display_comparison,
                                         display_feedback, interpret_feedback)
from recommender.input_module import collect_user_input
from recommender.intent_module import detect_intent, display_intent
from recommender.preference_module import analyze_preferences, display_analysis, load_catalog
from recommender.profile_module import build_profile, display_profile
from recommender.prompt_module import build_prompt, display_prompt
from recommender.ranking_module import display_ranking, rank_recommendations
from recommender.recommendation_module import display_recommendations, generate_recommendations
from recommender.save_module import save_recommendations

MENU = [
    ("1", "Enter user profile"),
    ("2", "View collected profile"),
    ("3", "Analyse preferences"),
    ("4", "Generate prompt"),
    ("5", "Get GPT-4 recommendations"),
    ("6", "Rank recommendations"),
    ("7", "Explain recommendations"),
    ("8", "Give feedback and refine"),
    ("9", "Chat with the assistant (intent detection)"),
    ("10", "Save recommendations"),
    ("11", "Run full pipeline (3 -> 7)"),
    ("0", "Exit"),
]


class Session:
    def __init__(self):
        self.profile = None
        self.analysis = None
        self.prompt = None
        self.recs = None
        self.ranked = None
        self.explained = None
        self.feedback_history = []

    # Each step computes what it needs from earlier steps if they have not run yet.
    def need_profile(self):
        if self.profile is None:
            ui.warn("No profile yet - please enter your details first.")
            self.input_profile()
        return self.profile

    def input_profile(self):
        self.profile = build_profile(collect_user_input())
        self.analysis = self.prompt = self.recs = self.ranked = self.explained = None
        ui.success(f"Profile created for {self.profile.name}.")

    def analyse(self):
        self.need_profile()
        ui.info("Analysing preferences with Sentence Transformers and the LLM...")
        self.analysis = analyze_preferences(self.profile)
        display_analysis(self.analysis)

    def make_prompt(self, feedback=None, previous=None):
        if self.analysis is None:
            self.analyse()
        self.prompt = build_prompt(self.profile, self.analysis, feedback, previous)
        display_prompt(self.prompt)

    def recommend(self):
        if self.prompt is None:
            self.make_prompt()
        ui.info(f"Querying {LLM_MODEL} for recommendations...")
        self.recs = generate_recommendations(self.prompt, self.analysis, self.profile.num_items)
        display_recommendations(self.recs)

    def rank(self):
        if self.recs is None:
            self.recommend()
        self.ranked = rank_recommendations(self.recs, self.profile)
        display_ranking(self.ranked, self.profile)

    def explain(self):
        if self.ranked is None:
            self.rank()
        ui.info("Generating personalised explanations...")
        self.explained = explain_recommendations(self.ranked, self.profile)
        display_explanations(self.explained)

    def refine(self, feedback=None):
        if self.ranked is None:
            self.rank()
        feedback = feedback or collect_feedback()
        if not feedback:
            ui.warn("No feedback entered.")
            return
        before = self.ranked
        parsed = interpret_feedback(feedback)
        apply_feedback(self.profile, parsed)
        display_feedback(feedback, parsed, self.profile)
        self.feedback_history.append({"feedback": feedback, **parsed})

        self.analysis = analyze_preferences(self.profile)
        self.prompt = build_prompt(self.profile, self.analysis, feedback,
                                   before["title"].tolist())
        self.recs = generate_recommendations(self.prompt, self.analysis, self.profile.num_items)
        self.ranked = rank_recommendations(self.recs, self.profile)
        self.explained = None
        display_comparison(before, self.ranked)

    def save(self):
        if self.ranked is None:
            self.rank()
        if self.explained is None or len(self.explained) != len(self.ranked):
            self.explained = explain_recommendations(self.ranked, self.profile)
        save_recommendations(self.profile, self.explained, self.feedback_history)

    def chat(self):
        self.need_profile()
        message = input("  Ask me anything (e.g. 'these are too easy'): ").strip()
        if not message:
            return True
        intent, score, ranked = detect_intent(message)
        display_intent(message, intent, score, ranked)
        actions = {
            "get_recommendations": self.rank,
            "refine_recommendations": lambda: self.refine(message),
            "change_skill_level": lambda: self.refine(message),
            "explain_recommendation": self.explain,
            "save_recommendations": self.save,
            "view_profile": lambda: display_profile(self.profile),
        }
        if intent == "exit":
            return False
        if intent in actions:
            ui.info(f"Action: {intent.replace('_', ' ')}")
            actions[intent]()
        else:
            ui.warn("Sorry, I could not understand that request.")
        return True


def home_screen():
    ui.banner("PERSONALIZED RECOMMENDATION SYSTEM USING GPT-4")
    print("  24CS2506 - Building Applications using GPT-4  |  Experiment 4")
    print(f"  LLM: {LLM_MODEL} (Ollama)   Embeddings: {EMBEDDING_MODEL}")
    catalog, _ = load_catalog()
    print(f"  Catalog: {len(catalog)} items across {catalog['domain'].nunique()} domains")
    print("\n  MAIN MENU")
    for key, label in MENU:
        print(f"   {key:>3}. {label}")


def main():
    session = Session()
    while True:
        home_screen()
        choice = input("\n  Enter your choice: ").strip()
        if choice == "0":
            break
        handlers = {
            "1": session.input_profile,
            "2": lambda: display_profile(session.need_profile()),
            "3": session.analyse,
            "4": session.make_prompt,
            "5": session.recommend,
            "6": session.rank,
            "7": session.explain,
            "8": session.refine,
            "10": session.save,
            "11": lambda: (session.analyse(), session.make_prompt(),
                           session.recommend(), session.rank(), session.explain()),
        }
        if choice == "9":
            if not session.chat():
                break
        elif choice in handlers:
            handlers[choice]()
        else:
            ui.warn("Invalid choice.")
    ui.banner("Thank you for using the Personalized Recommendation System!")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)
