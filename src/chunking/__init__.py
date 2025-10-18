# src/chunking/__init__.py
"""
This package contains various text chunking strategies.
The STRATEGIES dictionary maps strategy names (used in .env) to their
corresponding chunking functions, providing a clean way to select a method.
"""
from . import recursive_character
from . import token_based
from . import semantic_similarity
from . import llm_window
from . import propositional

# A mapping from strategy names (string) to the actual functions.
# This makes it easy to add new strategies and select them from config.
STRATEGIES = {
    "recursive_char": recursive_character.chunk,
    "token": token_based.chunk,
    "semantic_similarity": semantic_similarity.chunk,
    "llm_window": llm_window.chunk,
    "propositional": propositional.chunk,
}

DEFAULT_STRATEGY = "recursive_char"

def get_chunking_strategy(strategy_name: str):
    """
    Retrieves the chunking function based on its name.
    Falls back to the default strategy if the name is not found.
    """
    if strategy_name not in STRATEGIES:
        print(f"Warning: Chunking strategy '{strategy_name}' not found. "
              f"Falling back to '{DEFAULT_STRATEGY}'.")
        strategy_name = DEFAULT_STRATEGY
    return STRATEGIES[strategy_name]
