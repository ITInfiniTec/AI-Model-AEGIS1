# cognitive_fallacy_library.py

from typing import List, Dict
from config_loader import config_loader

class CognitiveFallacyLibrary:
    def __init__(self):
        # Load fallacies from the centralized configuration source.
        self.fallacies: Dict[str, List[str]] = config_loader.get_fallacies()

    def check_for_fallacies(self, text: str) -> List[str]:
        """Checks for cognitive fallacies in the given text."""
        detected_fallacies = []
        lower_text = text.lower()
        for fallacy, keywords in self.fallacies.items():
            if any(keyword in lower_text for keyword in keywords):
                detected_fallacies.append(fallacy)
        return detected_fallacies

cognitive_fallacy_library = CognitiveFallacyLibrary()