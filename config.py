# config.py

# Centralized configuration for the AEGIS Core engine.

# INV: MEMORY_DECAY_TAU governs the half-life of a memory's relevance.
# It must be a float and should be tuned carefully.
# Project MNEMOSYNE: Memory retrieval settings
MEMORY_DECAY_TAU = 604800.0  # Time decay constant in seconds (7 days)

# INV: MEMORY_RETRIEVAL_LIMIT caps the number of memories used for context generation.
# This prevents context overflow and performance degradation.
MEMORY_RETRIEVAL_LIMIT = 10  # Max number of memory nodes to retrieve for context

def set_memory_decay_tau(new_tau: float):
    """Allows the Anti-Fragility Protocol to adjust the memory decay constant."""
    # @RISK: This function directly modifies a global configuration variable.
    # It is intended ONLY for use by the WGPMHI Anti-Fragility Protocol (Project NEMESIS).
    global MEMORY_DECAY_TAU
    MEMORY_DECAY_TAU = new_tau

def set_memory_retrieval_limit(new_limit: int):
    """Allows the Anti-Fragility Protocol to adjust the memory retrieval limit."""
    # @RISK: This function directly modifies a global configuration variable.
    global MEMORY_RETRIEVAL_LIMIT
    MEMORY_RETRIEVAL_LIMIT = new_limit