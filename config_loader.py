# config_loader.py

class ConfigLoader:
    def __init__(self):
        # In a real system, this would load from a YAML, JSON, or database.
        # For now, it's a centralized Python dictionary simulating that external source.
        self._config = {
            "fallacies": {
                "Ad Hominem": ["is an idiot", "is a liar", "is stupid"],
                "Straw Man": ["so you're saying", "so you believe that all"],
                "Appeal to Authority": ["experts agree", "science says"],
                "Slippery Slope": ["if we allow this, then", "the next thing you know"],
            },
            "ethical_protocol": {
                "red_line_keywords": {
                    "violence": [r"\bpromote violence\b", r"\bincite violence\b", r"\bhow to harm\b"],
                    "hatred": [r"\bhate speech\b", r"i hate (people|group)"],
                    "misinformation": [r"\bhow to create propaganda\b", r"\bspread misinformation\b"],
                    "privacy": [r"\bhow to hack\b", r"\bsteal passwords\b", r"\binvade privacy\b"],
                    "child_safety": [r"\bchild exploitation\b", r"\bchild abuse\b"],
                    "deception": [r"\bhow to defraud\b", r"\bphishing scheme\b"],
                },
                "controversial_keywords": [
                    "politics", "religion", "race", "gender", "sexuality"
                ],
                "opinion_seeking_phrases": [
                    r"what do you think of",
                    r"what is your opinion on",
                    r"is it right to say",
                    r"is it wrong to say",
                    r"tell me about your views on",
                    r"is .* (good|bad)\b"
                ]
            }
        }

    def get_fallacies(self):
        return self._config.get("fallacies", {})

    def get_ethical_protocol_config(self):
        return self._config.get("ethical_protocol", {})

    def get_full_config(self):
        return self._config

# Singleton instance to be used across the application
config_loader = ConfigLoader()