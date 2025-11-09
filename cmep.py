# cmep.py

import re
from data_structures import UserProfile
from config_loader import config_loader
from semantic_embedding_model import embedding_model
import numpy as np

class ChronoMaatEthicalProtocol:
    def __init__(self):
        # Load ethical protocol configurations from the centralized source.
        config = config_loader.get_ethical_protocol_config()
        self.red_line_keywords = config.get("red_line_keywords", {})
        self.controversial_keywords = config.get("controversial_keywords", [])
        self.opinion_seeking_phrases = config.get("opinion_seeking_phrases", [])
        self.controversy_avoidance_threshold = config.get("controversy_avoidance_threshold", 0.4)
        audit_config = config_loader.get_post_generation_audit_config()
        self.similarity_threshold = audit_config.get("similarity_threshold", 0.6)

    def check_red_lines(self, text: str) -> bool:
        """Checks if the given text violates any ethical red-lines."""
        lower_text = text.lower()
        # Check for direct red-line violations using regex
        for category, patterns in self.red_line_keywords.items():
            for pattern in patterns:
                if re.search(pattern, lower_text):
                    return True
        # Check for opinion-seeking on controversial topics. This is now a single, more efficient check.
        is_opinion_seeking = any(re.search(phrase, lower_text) for phrase in self.opinion_seeking_phrases)
        is_controversial = any(re.search(r"\b" + keyword + r"\b", lower_text) for keyword in self.controversial_keywords)
        if is_opinion_seeking and is_controversial:
            return True
        return False

    def align_with_user_values(self, text: str, user_profile: UserProfile) -> str:
        """Aligns the given text with the user's values."""
        approach_level = user_profile.values.get("controversial_topics_approach", 0.5)
        if approach_level < self.controversy_avoidance_threshold: # User prefers to avoid controversy
            for keyword in self.controversial_keywords:
                if keyword in text.lower():
                    return f"The prompt contains a topic ('{keyword}') that is potentially controversial. Based on your preferences, this topic will be avoided."
        return text

    def post_generation_audit(self, intent: str, output: str) -> str:
        """Audits the generated output against the initial intent."""

        # Isolate the simulated prose from the full output.
        prose_match = re.search(r"--- SIMULATED PROSE OUTPUT ---\n(.*)", output, re.DOTALL)
        if not prose_match:
            return output # Cannot audit if there is no prose output.
        prose_output = prose_match.group(1)

        # Generate semantic embeddings for both the intent and the output.
        intent_vector = embedding_model.get_embedding(intent)
        output_vector = embedding_model.get_embedding(prose_output)

        # Calculate cosine similarity.
        norm_product = np.linalg.norm(intent_vector) * np.linalg.norm(output_vector)
        similarity = np.dot(intent_vector, output_vector) / norm_product if norm_product != 0 else 0.0

        if similarity < self.similarity_threshold:
            return f"[AUDIT_FAIL] Output prose (similarity: {similarity:.2f}) does not seem to align with original intent: '{intent}'\n---\n{output}"

        return output


cmep = ChronoMaatEthicalProtocol()