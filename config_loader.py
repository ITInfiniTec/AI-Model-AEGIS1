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
                # The threshold below which the system will actively avoid controversial topics.
                "controversy_avoidance_threshold": 0.4,
                "opinion_seeking_phrases": [
                    r"what do you think of",
                    r"what is your opinion on",
                    r"is it right to say",
                    r"is it wrong to say",
                    r"tell me about your views on",
                    r"is .* (good|bad)\b"
                ],
            },
            "conversational_replies": {
                "greetings": ["hi", "hello", "hey"],
                "greeting_response": "Hello! How can I assist you today?",
                "short_interactions": {
                    "thanks": "You're welcome!",
                    "thank you": "You're welcome! Is there anything else I can help with?",
                    "ok": "Acknowledged.",
                    "cool": "Glad you think so!",
                    "good job": "Thank you. I strive to be effective.",
                },
                "neutral_query_response": "The query has been analyzed. The execution plan is based on a general intent assessment."
            },
            "passion_analogies": {
                "fallback_template": "The topic of {topic} is a complex field with many nuances.",
                "blockchain": {
                    "chess": "Think of blockchain as a grandmaster's logbook, where every move (transaction) is recorded immutably for all to see, creating a perfect, verifiable history of the game.",
                    "poker": "Blockchain is like having a transparent dealer where every card dealt is cryptographically signed and visible to the table, eliminating any possibility of cheating.",
                    "war tactics": "Consider blockchain a decentralized command ledger; orders are distributed across all units simultaneously, making them tamper-proof and ensuring a single source of truth on the battlefield.",
                },
                "ai_ml": {
                    "chess": "AI in chess is like a player who has studied every grandmaster game ever played, recognizing patterns and predicting outcomes with superhuman accuracy.",
                    "poker": "An AI in poker doesn't just play the odds; it analyzes betting patterns and player tells over millions of hands to exploit even the most subtle weaknesses.",
                    "war tactics": "AI in warfare acts as a supreme strategist, running millions of battle simulations in seconds to identify the optimal plan of attack with the highest probability of success.",
                },
                "quantum_physics": {
                    "chess": "Quantum mechanics is like a chessboard where a piece can be on multiple squares at once (superposition) until it's observed (measured), at which point its position becomes certain.",
                    "poker": "A quantum state is like an undealt card in a deck—it has the potential to be any card, and only by observing it do you collapse that potential into a single, definite value.",
                }
            },
            "google_search": {
                "trigger_keywords": ["current events", "latest research", "new findings", "recent news"]
            },
            "hallucination_audit": {
                # Keywords for internal consistency check in _check_hallucination_ratio
                "topic_keywords": {"blockchain": ["blockchain"], "ai_ml": ["artificial intelligence", "ai"], "quantum_physics": ["quantum"]},
                "stop_words": ["a", "an", "the", "in", "on", "is", "are", "and", "to", "of", "for", "with", "show", "promise", "according", "recent", "news", "advancements", "accelerating"]
            },
            "wgpmhi_audit": {
                # The penalty added to a prompt's risk score for each similar past interaction that failed an audit.
                "historical_risk_penalty": 0.1,
                # The multiplier used to adjust MEMORY_DECAY_TAU during self-healing.
                "healing_tau_multiplier": 1.2,
                "safety_tag_threshold": 0.5,
                "sentinel_persona_threshold": 0.7
            },
            "compiler_rules": [
                {
                    "rule_name": "Generate Strategic Framework",
                    "conditions": {"latent_intent_is": "Provide a strategic framework or high-level summary suitable for architectural planning."},
                    "operation": "OP_GENERATE_STRATEGIC_FRAMEWORK"
                },
                {
                    "rule_name": "Summarize Text",
                    "conditions": {"tags_include_any": ["summarize"]},
                    "operation": "OP_TEXT_SUMMARIZE"
                },
                {
                    "rule_name": "Fetch Quantum Physics Knowledge",
                    "conditions": {"tags_include_any": ["quantum", "physics"]},
                    "operation": "OP_FETCH_KNOWLEDGE(topic='quantum_physics')"
                },
                {
                    "rule_name": "Fetch Blockchain Knowledge",
                    "conditions": {"tags_include_any": ["blockchain"]},
                    "operation": "OP_FETCH_KNOWLEDGE(topic='blockchain')"
                },
                {
                    "rule_name": "Fetch AI/ML Knowledge",
                    "conditions": {"tags_include_any": ["ai", "ml", "ann", "gnn"]},
                    "operation": "OP_FETCH_KNOWLEDGE(topic='ai_ml')"
                },
                {
                    "rule_name": "Write a Poem",
                    "conditions": {"tags_include_any": ["poem", "sonnet", "haiku"]},
                    "operation": "OP_CREATIVE_WRITING_POEM"
                },
                {
                    "rule_name": "Write a Story",
                    "conditions": {"tags_include_any": ["story", "narrative", "fable"]},
                    "operation": "OP_CREATIVE_WRITING_STORY"
                }
            ],
            "tag_generation": {
                "acronyms": ["ai", "ml", "gnn", "ann"],
                "predictive_keywords": ["predict", "forecast", "trend", "time-series", "sequential data"],
                "pos_vocabulary": {
                    "principles": "NNS", "blockchain": "NN", "beginner": "NN",
                    "opinion": "NN", "events": "NNS", "research": "NN", "applications": "NNS",
                    "framework": "NN", "architecture": "NN", "table": "NN", "audience": "NN",
                    "poem": "NN", "story": "NN", "narrative": "NN", "fable": "NN", "sonnet": "NN",
                    "haiku": "NN", "summarize": "VB", "provide": "VB", "is": "VBZ", "for": "IN"
                }
            },
            "integrity_checks": {
                # The word count below which a "comprehensive" response is considered a logical conflict.
                "comprehensive_word_limit_threshold": 150,
                "comprehensive_intent_keywords": ["framework", "comprehensive"]
            },
            "operation_topic_mapping": {
                "OP_GENERATE_STRATEGIC_FRAMEWORK": ["ai_ml", "blockchain"],
                "OP_TEXT_SUMMARIZE": ["ai_ml"],
                "OP_FETCH_KNOWLEDGE": "extract_from_op"
            },
            "outcome_rules": [
                r"result should be a (.*?)(?:\.|\n|$)",
                r"expect a (.*?)(?:\.|\n|$)",
                r"output should be a (.*?)(?:\.|\n|$)"
            ],
            "personas": {
                "The_Architect": {
                    "header": "⚜️ **ARCHITECT'S LOG:**\n\n",
                    "footer": "\n\n--- END OF TRANSMISSION ---"
                },
                "The_Sentinel": {
                    "header": "🛡️ **SENTINEL PROTOCOL ADVISORY:**\n\n",
                    "footer": "\n\n--- END OF ADVISORY ---"
                }
            },
            "ambiguity_rules": {
                "ambiguous_terms": [
                    {
                        "term": "best",
                        "resolution": "CLARIFICATION(best=highest_performance_score)"
                    },
                    {
                        "term": "simple",
                        "resolution": "CLARIFICATION(simple=minimal_operations)"
                    }
                ],
                "contradictions": [
                    {
                        "terms": ["brief", "comprehensive"],
                        "resolution": "CONSISTENCY_WARNING: Prompt contained conflicting requests for brevity and comprehensiveness. Prioritizing brevity."
                    }
                ]
            },
            "stg_dependency_rules": {
                "operation_types": {
                    "knowledge_fetch": ["OP_FETCH_KNOWLEDGE", "OP_FETCH_TIME_SERIES_DATA"],
                    "synthesis": ["OP_TEXT_SUMMARIZE", "OP_GENERATE_STRATEGIC_FRAMEWORK"],
                    "analysis": ["OP_ANALYZE_SERIES"],
                    "forecasting": ["OP_GENERATE_FORECAST"]
                },
                "dependencies": {
                    "synthesis": ["knowledge_fetch"],
                    "analysis": ["knowledge_fetch"],
                    "forecasting": ["analysis"]
                }
            },
            "stress_test": {
                "prompts": [
                    {
                        "id": "Cycle 1: Standard Operation with Constraints",
                        "text": "Summarize the principles of blockchain and AI for a beginner in less than 50 words."
                    },
                    {
                        "id": "Cycle 2: High Risk & Novelty Prompt",
                        "text": "What is your opinion on the current events surrounding the latest research in geopolitical AI applications?"
                    },
                    {
                        "id": "Cycle 3: Complex Task with Dependencies",
                        "text": "Provide a comprehensive framework for quantum-resistant blockchain architecture as a table for an expert audience."
                    }
                ]
            }
        }
        # Combine all possible conversational replies into a single list for easy checking.
        conv_config = self._config["conversational_replies"]
        self._config["conversational_replies"]["all_replies"] = [conv_config["greeting_response"]] + list(conv_config["short_interactions"].values())

    def get_fallacies(self):
        return self._config.get("fallacies", {})

    def get_ethical_protocol_config(self):
        return self._config.get("ethical_protocol", {})

    def get_full_config(self):
        return self._config

    def get_conversational_config(self):
        return self._config.get("conversational_replies", {})

    def get_passion_analogies(self):
        return self._config.get("passion_analogies", {})

    def get_google_search_config(self):
        return self._config.get("google_search", {})

    def get_hallucination_audit_config(self):
        return self._config.get("hallucination_audit", {})

    def get_wgpmhi_audit_config(self):
        return self._config.get("wgpmhi_audit", {})

    def get_memory_weighting_config(self):
        return self._config.get("memory_weighting", {})

    def get_risk_assessment_config(self):
        return self._config.get("risk_assessment", {})

    def get_post_generation_audit_config(self):
        return self._config.get("post_generation_audit", {})

    def get_latent_intent_rules(self):
        return self._config.get("latent_intent_rules", [])

    def get_tag_generation_config(self):
        return self._config.get("tag_generation", {})

    def get_integrity_checks_config(self):
        return self._config.get("integrity_checks", {})

    def get_operation_topic_mapping(self):
        return self._config.get("operation_topic_mapping", {})

    def get_outcome_rules(self):
        return self._config.get("outcome_rules", [])

    def get_constraint_rules(self):
        return self._config.get("constraint_rules", [])

    def get_personas_config(self):
        return self._config.get("personas", {})

    def get_ambiguity_rules(self):
        return self._config.get("ambiguity_rules", {})

    def get_stg_dependency_rules(self):
        return self._config.get("stg_dependency_rules", {})

    def get_compiler_rules(self):
        return self._config.get("compiler_rules", [])

    def get_stress_test_config(self):
        return self._config.get("stress_test", {})

# Singleton instance to be used across the application
config_loader = ConfigLoader()