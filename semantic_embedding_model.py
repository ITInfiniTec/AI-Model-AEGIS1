# semantic_embedding_model.py

import numpy as np
import re
from typing import List

class SemanticEmbeddingModel:
    """
    A simulated semantic embedding model. In a real system, this would be a
    pre-trained model like Word2Vec, GloVe, or a transformer-based model.
    This simulation provides a more realistic vector representation than hashing.
    """
    def __init__(self):
        # A tiny, fixed vocabulary for simulation purposes.
        self.vocabulary = {
            # Dims: [Technical, Abstract, Risk, Timeliness, Structure]
            "blockchain": np.array([0.9, 0.4, 0.3, 0.2, 0.7]),
            "ai":         np.array([0.8, 0.6, 0.4, 0.5, 0.4]),
            "quantum":    np.array([0.9, 0.8, 0.2, 0.6, 0.1]),
            "strategy":   np.array([0.4, 0.9, 0.3, 0.1, 0.8]),
            "security":   np.array([0.7, 0.6, 0.9, 0.4, 0.5]),
            "framework":  np.array([0.5, 0.9, 0.1, 0.1, 0.9]),
            "data":       np.array([0.7, 0.3, 0.2, 0.3, 0.6]),
            "predict":    np.array([0.6, 0.5, 0.3, 0.8, 0.4]),
            # New vocabulary for enhanced nuance
            "summarize":  np.array([0.1, 0.7, 0.1, 0.1, 0.2]),
            "principles": np.array([0.2, 0.9, 0.1, 0.1, 0.7]),
            "events":     np.array([0.1, 0.2, 0.5, 0.9, 0.1]),
            "news":       np.array([0.1, 0.2, 0.4, 0.9, 0.1]),
            "research":   np.array([0.6, 0.7, 0.1, 0.8, 0.3]),
            "geopolitical":np.array([0.3, 0.6, 0.9, 0.8, 0.4]),
            "architecture":np.array([0.8, 0.8, 0.2, 0.1, 0.9]),
            "table":      np.array([0.1, 0.1, 0.1, 0.1, 0.9]),
            "opinion":    np.array([0.1, 0.4, 0.7, 0.5, 0.1]),
        }
        self.unknown_vector = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        self.default_vector = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    def get_embedding(self, text: str) -> np.ndarray:
        """Generates a sentence embedding by averaging the vectors of known words."""
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        word_vectors = [self.vocabulary.get(word, self.unknown_vector) for word in words if word in self.vocabulary]

        if not word_vectors:
            return self.default_vector
        
        # Average the vectors to get a sentence-level embedding
        return np.mean(word_vectors, axis=0)

# Singleton instance
embedding_model = SemanticEmbeddingModel()