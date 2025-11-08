# config.py

# Centralized configuration for the AEGIS Core engine.

# Project MNEMOSYNE: Memory retrieval settings
MEMORY_DECAY_TAU = 604800.0  # Time decay constant in seconds (7 days)
MEMORY_RETRIEVAL_LIMIT = 10  # Max number of memory nodes to retrieve for context

def set_memory_decay_tau(new_tau: float):
    """Allows the Anti-Fragility Protocol to adjust the memory decay constant."""
    global MEMORY_DECAY_TAU
    MEMORY_DECAY_TAU = new_tau

def set_memory_retrieval_limit(new_limit: int):
    """Allows the Anti-Fragility Protocol to adjust the memory retrieval limit."""
    global MEMORY_RETRIEVAL_LIMIT
    MEMORY_RETRIEVAL_LIMIT = new_limit