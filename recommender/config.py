"""Central configuration for the recommendation system."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = BASE_DIR / "data" / "catalog.csv"
OUTPUT_DIR = BASE_DIR / "output"

# Local LLM served by Ollama (stands in for GPT-4 in the lab setup).
LLM_MODEL = os.getenv("RECSYS_LLM_MODEL", "llama3.2")
LLM_TEMPERATURE = 0.3

# Sentence-Transformers model used for semantic matching and intent detection.
EMBEDDING_MODEL = os.getenv("RECSYS_EMBED_MODEL", "all-MiniLM-L6-v2")

# Number of catalog candidates retrieved before the LLM picks the final list.
CANDIDATE_POOL = 15

SKILL_LEVELS = ["Beginner", "Intermediate", "Advanced"]
CATEGORIES = ["Course", "Project", "Book", "Certification", "Any"]
