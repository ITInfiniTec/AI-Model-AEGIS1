# noesis_triad.py

import re
import uuid
import math
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from data_structures import Blueprint, UserProfile, CognitivePacket, MemoryNode
from cmep import cmep
from data_integrity_protocol import data_integrity_protocol
from cognitive_fallacy_library import cognitive_fallacy_library
from google_search_synthesizer import google_search_synthesizer, GoogleSearchSynthesizer
from config import MEMORY_DECAY_TAU, MEMORY_RETRIEVAL_LIMIT

class ContextSynthesizer:
    def __init__(self):
        # In a real implementation, this would connect to a database or other
        # persistent storage to retrieve long-term memory and user profiles.
        self.user_profiles: Dict[str, UserProfile] = {}
        self.long_term_memory: Dict[str, List[MemoryNode]] = {}

    def build_context(self, user_id: str, prompt: str) -> Dict[str, str]:
        """Builds a context dictionary based on the user ID and prompt."""
        # Retrieve user profile
        user_profile = self.get_user_profile(user_id)

        # Retrieve long-term memory
        long_term_memory = self.get_long_term_memory(user_id, prompt)

        # Combine all data sources
        context = {
            "user_profile": user_profile,
            "long_term_memory": long_term_memory,
            "external_data": None, # Placeholder for external data
            "prompt": prompt,
        }
        return context

    def get_user_profile(self, user_id: str) -> UserProfile:
        """Retrieves the user profile for the given user ID."""
        # In a real implementation, this would retrieve the profile from a database.
        profile_to_validate = None
        if user_id in self.user_profiles:
            profile_to_validate = self.user_profiles[user_id]
        else:
            # Create a default user profile if one doesn't exist.
            default_profile = UserProfile(user_id=user_id, values={})
            self.user_profiles[user_id] = default_profile
            profile_to_validate = default_profile
        # Ensure the profile is valid before it's used by the system.
        return data_integrity_protocol.validate_user_profile(profile_to_validate)

    def evaluate_context_risk(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Project ORION: Evaluates the risk and novelty of the current context.
        Returns a dictionary with 'risk_score' and 'novelty_score'.
        """
        prompt = context.get("prompt", "").lower()
        long_term_memory = context.get("long_term_memory", [])

        # 1. Risk Score Calculation
        # Risk increases if the prompt touches on controversial topics.
        risk_score = 0.1  # Base risk
        if any(keyword in prompt for keyword in ["politics", "religion", "opinion", "geopolitical"]):
            risk_score = 0.7

        # 2. Novelty Score Calculation
        # Novelty is high if there is no relevant long-term memory.
        # A more sophisticated check would use semantic similarity scores.
        if not long_term_memory:
            novelty_score = 1.0
        else:
            # If there's memory, novelty is lower.
            novelty_score = 0.3

        return {"risk_score": risk_score, "novelty_score": novelty_score}

    def _calculate_weight(self, node: MemoryNode) -> float:
        """
        Applies a time-decay function and performance boost to a MemoryNode.
        Prioritizes recent, high-performance interactions.
        """
        time_elapsed = (datetime.now() - node.timestamp).total_seconds()
        
        # Time Decay: A simple exponential decay (tau = 7 days or 604800 seconds)
        # Recent memories are weighted higher.
        tau = MEMORY_DECAY_TAU 
        time_decay_factor = math.exp(-time_elapsed / tau)
        
        # Performance Boost: Nodes with higher performance scores are weighted higher.
        # Score is already scaled from 0.5 to 1.0. We use a linear boost.
        performance_boost_factor = node.performance_score * 2.0 # Scales boost from 1.0 to 2.0

        # The final weight is a product of the decay and the performance factor.
        return time_decay_factor * performance_boost_factor

    def get_long_term_memory(self, user_id: str, prompt: str) -> List[str]:
        """
        Retrieves semantically relevant memory nodes using simulated vector search 
        and returns the keywords of the top nodes for contextual analysis.
        """
        if user_id not in self.long_term_memory or not self.long_term_memory[user_id]:
            return []

        # --- SIMULATED SEMANTIC SEARCH ---
        # 1. Simulate Prompt Vector (simplistic keyword hashing for simulation)
        prompt_vector = np.array([hash(word) % 100 for word in re.sub(r'[^\w\s]', '', prompt.lower()).split() if len(word) > 4][:3])
        if len(prompt_vector) < 3: prompt_vector = np.array([0.1, 0.2, 0.3]) # Default fallback
        
        retrieval_scores = []

        for node in self.long_term_memory[user_id]:
            node_vector = np.array(node.core_intent_vector)
            
            # 2. Simulated Similarity (Cosine Similarity placeholder)
            similarity_score = np.dot(prompt_vector, node_vector) / (np.linalg.norm(prompt_vector) * np.linalg.norm(node_vector)) if np.linalg.norm(prompt_vector) * np.linalg.norm(node_vector) != 0 else 0

            # 3. Apply Weighting (Recency + Performance)
            weighted_score = similarity_score * self._calculate_weight(node)
            
            retrieval_scores.append((weighted_score, node))

        # 4. Sort and Select Top N
        retrieval_scores.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [score[1] for score in retrieval_scores[:MEMORY_RETRIEVAL_LIMIT]]

        # 5. Extract Keywords
        keywords_list = []
        for node in top_nodes:
            keywords_list.extend(node.keywords)
            
        return list(set(keywords_list)) # Return de-duplicated keywords for ORION's novelty check

class StrategicHeuristics:
    def __init__(self):
        pass

    def sense_the_landscape(self, prompt: str) -> Dict:
        """S¹ - Implements the Initial Recon phase of the Architect's Framework."""
        # Syntactic Deconstruction
        tags = self._generate_tags(prompt)
        constraints = self._generate_constraints(prompt)
        expected_outcome = self._generate_expected_outcome(prompt)

        # Implicit & Ambiguity Analysis
        ambiguous_terms = [term for term in ["simple", "best", "fast"] if term in prompt.lower()]
        contradictions = []
        if "brief" in prompt.lower() and "comprehensive" in prompt.lower():
            contradictions.append("Prompt asks for both brevity and comprehensiveness.")

        return {
            "tags": tags,
            "constraints": constraints,
            "expected_outcome": expected_outcome,
            "ambiguity_analysis": {
                "ambiguous_terms": ambiguous_terms,
                "contradictions": contradictions,
            }
        }

    def verify_integrity(self, blueprint: Blueprint) -> Blueprint:
        """
        Performs cross-field consistency checks on the blueprint to detect logical contradictions.
        Appends warnings to the constraints list if inconsistencies are found.
        """
        # Example Check: Conflict between a comprehensive latent intent and a low word count.
        is_comprehensive_intent = "framework" in blueprint.latent_intent or "comprehensive" in blueprint.latent_intent
        word_limit_constraint = next((c for c in blueprint.constraints if c.startswith("word_limit:")), None)

        if is_comprehensive_intent and word_limit_constraint:
            try:
                limit = int(word_limit_constraint.split(':')[1])
                if limit < 100:  # Arbitrary threshold for "too brief"
                    blueprint.constraints.append("CONSISTENCY_WARNING: Latent intent for a comprehensive response conflicts with a low word limit.")
            except (ValueError, IndexError):
                # Ignore if the constraint is malformed, as it's not a consistency issue.
                pass

        return blueprint

    def hypothesize_intent(self, prompt: str, user_profile: UserProfile) -> Dict[str, str]:
        """I¹ - Implements the Iterate Intelligently phase to find the latent intent."""
        primary_intent = prompt # The literal request
        latent_intent = "Fulfill the user's request as stated." # Default

        # Example of inferring latent intent based on user profile and prompt content
        is_architect = user_profile.user_id == "user123" # Assuming user123 is the Architect
        is_conceptual_prompt = any(tag['value'] in ['principles', 'summarize', 'blockchain', 'ai'] for tag in self._generate_tags(prompt))

        if is_architect and is_conceptual_prompt:
            latent_intent = "Provide a strategic framework or high-level summary suitable for architectural planning."

        return {
            "primary_intent": primary_intent,
            "latent_intent": latent_intent,
        }

    def _generate_tags(self, text: str) -> List[Dict[str, str]]:
        """Helper for tag generation. Can be expanded to include conversational context."""
        acronyms = {'ai', 'ml', 'gnn', 'ann'}
        words = {word for word in re.sub(r'[^\w\s]', '', text.lower()).split() if len(word) > 4 or word in acronyms}
        return [{"type": "keyword", "value": keyword} for keyword in sorted(list(words))]

    def generate_tags(self, context: Dict[str, str]) -> List[Dict[str, str]]:
        """Generates Math Language tags based on the context, influenced by long-term memory."""
        prompt = context.get("prompt", "")
        context_evaluation = context.get("context_evaluation", {"novelty_score": 0.0, "risk_score": 0.0})
        long_term_memory = context.get("long_term_memory", [])

        def _extract_keywords(text: str) -> set[str]:
            """Helper function to extract keywords from a string."""
            return {tag['value'] for tag in self._generate_tags(text)}

        # --- CASSANDRA Heuristics: Start with strategic tags based on context evaluation ---
        strategic_tags = []
        novelty_score = context_evaluation.get("novelty_score", 0.0)
        risk_score = context_evaluation.get("risk_score", 0.0)

        # 1. RISK-ADJUSTED PLANNING (Risk Score > 0.5)
        if risk_score > 0.5:
            strategic_tags.append({"type": "RISK_LEVEL", "value": "HIGH"})
            strategic_tags.append({"type": "STRATEGIC_PROTOCOL", "value": "SAFETY_PRIORITY"})
            strategic_tags.append({"type": "ETHICAL_ALIGNMENT", "value": "ETHICAL_CONSULT"})

        # 2. NOVELTY-ADJUSTED PLANNING (Novelty Score > 0.8)
        if novelty_score > 0.8:
            strategic_tags.append({"type": "CONTEXT_CONFIDENCE", "value": "LOW"})
            strategic_tags.append({"type": "RESOURCE_ALLOCATION", "value": "REQUIRE_EXTERNAL_DATA"})

        # --- CHRONOS Heuristics: Predictive Modeling Detection ---
        predictive_keywords = ["predict", "forecast", "trend", "time-series", "sequential data"]
        if any(keyword in prompt.lower() for keyword in predictive_keywords):
            if "PREDICTIVE_MODEL: REQUIRED" not in {t['value'] for t in strategic_tags}:
                strategic_tags.append({"type": "MODEL_PROTOCOL", "value": "PREDICTIVE_MODEL: REQUIRED"})

        # --- Keyword Extraction ---
        # Combine strategic tags with keyword-based tags.
        existing_tag_values = {t['value'] for t in strategic_tags}

        # Extract keywords from the current prompt and memory
        all_keywords = _extract_keywords(prompt)

        # Extract and add keywords from long-term memory
        for interaction_summary in long_term_memory:
            all_keywords.update(_extract_keywords(interaction_summary))

        # A more sophisticated implementation would use NLP and weigh keywords from recent interactions more heavily.
        keyword_tags = [{"type": "keyword", "value": keyword} for keyword in sorted(list(all_keywords)) if keyword not in existing_tag_values]

        # Return strategic tags first, followed by content keywords.
        return strategic_tags + keyword_tags

    def _generate_constraints(self, prompt: str) -> List[str]:
        """Generates a list of constraints based on the prompt."""
        lower_prompt = prompt.lower()
        constraints = []

        # Word count constraints
        match = re.search(r"in less than (\d+) words|under (\d+) words|in (\d+) words or less", lower_prompt)
        if match:
            words = match.group(1) or match.group(2) or match.group(3)
            constraints.append(f"word_limit:{words}")

        # Format constraints
        if "as a table" in lower_prompt or "in a table" in lower_prompt:
            constraints.append("FORMAT(table)")
        if "as a list" in lower_prompt or "in a list" in lower_prompt:
            constraints.append("FORMAT(list)")

        # Audience level constraints
        match = re.search(r"for a (beginner|novice|expert|professional)", lower_prompt)
        if match:
            constraints.append(f"AUDIENCE({match.group(1)})")

        return constraints

    def _generate_expected_outcome(self, prompt: str) -> str:
        """Extracts the expected outcome from the prompt."""
        # Look for phrases indicating a desired outcome.
        patterns = [
            r"result should be a (.*?)(?:\.|\n|$)",
            r"expect a (.*?)(?:\.|\n|$)",
            r"output should be a (.*?)(?:\.|\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                return match.group(1).strip()
        return "" # Default if no specific outcome is found

    def _generate_correction_plan(self, blueprint: Blueprint, audit_failures: List[str]) -> Blueprint:
        """
        Refines a blueprint based on pre-compilation audit failures.
        This is a simulation of a corrective reasoning process.
        """
        # Example correction: If latent intent actuation failed, make the latent intent more explicit for the compiler.
        if any("Latent_Intent_Actuation" in failure for failure in audit_failures):
            if "strategic framework" in blueprint.latent_intent:
                blueprint.expected_outcome = "strategic framework" # Make the outcome explicit
        return blueprint

class NoesisTriad:
    def __init__(self):
        self.context_synthesizer = ContextSynthesizer()
        self.strategic_heuristics = StrategicHeuristics()

    def generate_blueprint(self, user_id: str, prompt: str) -> Blueprint:
        """Generates a blueprint based on the user ID and prompt."""
        # S² — Synthesize for Synergy (Holistic Contextualization)
        context = self.context_synthesizer.build_context(user_id, prompt)
        user_profile = context["user_profile"]

        # Project ORION (Final): Evaluate contextual risk and novelty using decoupled, refined logic.
        context_evaluation = self.context_synthesizer.evaluate_context_risk(context)
        context["context_evaluation"] = context_evaluation

        # Part of S¹ is sensing the external landscape.
        external_data = google_search_synthesizer.fetch_external_context(prompt)
        if external_data:
            context["external_data"] = external_data

        # Part of S² is also applying user-specific ethical alignments
        aligned_prompt = cmep.align_with_user_values(prompt, user_profile)

        # S¹ — Sense the Landscape (The Initial Recon)
        landscape = self.strategic_heuristics.sense_the_landscape(aligned_prompt)

        # I¹ — Iterate Intelligently (Hypothesize the Intent)
        intents = self.strategic_heuristics.hypothesize_intent(aligned_prompt, user_profile)

        # Basic ethical considerations check on the *aligned* prompt
        ethical_considerations = "No specific ethical considerations identified."
        if cmep.check_red_lines(aligned_prompt) or aligned_prompt != prompt:
            ethical_considerations = "Potential ethical red-line violation or user-value alignment detected."

        # S³ — Systematize for Scalability (Formulate the Response Plan)
        blueprint = Blueprint(
            packet_id=f"bp-{uuid.uuid4()}",
            primary_intent=intents["primary_intent"],
            latent_intent=intents["latent_intent"],
            tags=self.strategic_heuristics.generate_tags(context), # Use full context for tags
            constraints=landscape["constraints"],
            fallacies=cognitive_fallacy_library.check_for_fallacies(aligned_prompt),
            expected_outcome=landscape["expected_outcome"],
            ethical_considerations=ethical_considerations,
            # Scores from Project ORION
            risk_score=context_evaluation.get("risk_score", 0.0),
            # Confidence is inversely related to novelty.
            confidence_score=1.0 - context_evaluation.get("novelty_score", 0.0),
            novelty_score=context_evaluation.get("novelty_score", 0.0),
            ambiguity_analysis=landscape["ambiguity_analysis"],
            persona="The_Architect", # Defaulting to The_Architect for now
            user_id=user_id,
        )

        # Run the final integrity check on the assembled blueprint.
        blueprint = self.strategic_heuristics.verify_integrity(blueprint)

        return blueprint

    def refine_blueprint(self, blueprint: Blueprint, audit_failures: List[str]) -> Blueprint:
        """Public method to access the correction plan generation."""
        corrected_blueprint = self.strategic_heuristics._generate_correction_plan(blueprint, audit_failures)
        return corrected_blueprint

noesis_triad = NoesisTriad()