# noesis_triad.py

import re
import uuid
import math
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Union
from data_structures import Blueprint, UserProfile, CognitivePacket, MemoryNode
from cmep import cmep
from data_integrity_protocol import data_integrity_protocol
from cognitive_fallacy_library import cognitive_fallacy_library
from google_search_synthesizer import google_search_synthesizer, GoogleSearchSynthesizer
from state_manager import state_manager
from semantic_embedding_model import embedding_model
from logger import log
from config import MEMORY_DECAY_TAU, MEMORY_RETRIEVAL_LIMIT

class ContextSynthesizer:
    def __init__(self):
        # These dictionaries act as a session-level cache for performance.
        # They are populated from the StateManager when a user is first encountered.
        self.user_profiles: Dict[str, UserProfile] = {}
        self.long_term_memory: Dict[str, List[MemoryNode]] = {}

    def build_context(self, user_id: str, prompt: str) -> Dict[str, str]:
        """Builds a context dictionary based on the user ID and prompt."""
        # Retrieve user profile
        user_profile = self.get_user_profile(user_id)

        # Retrieve long-term memory
        memory_keywords, memory_scores = self.get_long_term_memory(user_id, prompt)

        # Combine all data sources
        context = {
            "user_profile": user_profile,
            "long_term_memory": memory_keywords,
            # Pass the similarity scores for novelty calculation
            "memory_scores": memory_scores,
            "external_data": None, # Placeholder for external data
            "prompt": prompt,
        }
        return context

    def get_user_profile(self, user_id: str) -> UserProfile:
        """Retrieves the user profile for the given user ID."""
        profile_to_validate = None
        if user_id in self.user_profiles:
            profile_to_validate = self.user_profiles[user_id]
        else:
            # On first encounter in a session, load from StateManager or create a default.
            # This is a placeholder for loading a saved profile.
            default_profile = UserProfile(user_id=user_id, values={})
            log.info(f"Creating new in-memory session for user '{user_id}'.")
            self.user_profiles[user_id] = default_profile
            profile_to_validate = default_profile
        # Ensure the profile is valid before it's used by the system.
        return data_integrity_protocol.validate_user_profile(profile_to_validate)

    def evaluate_context_risk(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Project ORION: Evaluates the risk and novelty of the current context.
        Returns a dictionary with 'risk_score' and 'novelty_score'.
        """
        prompt = context.get("prompt", "")
        memory_scores = context.get("memory_scores", [])

        # 1. Risk Score Calculation
        # Use the semantic embedding model for a more nuanced risk assessment.
        # The 3rd dimension (index 2) of our embedding represents "Risk/Controversy".
        prompt_embedding = embedding_model.get_embedding(prompt)
        # The risk score is the normalized value of the risk dimension.
        # We add a small base risk and scale it.
        risk_score = 0.1 + (prompt_embedding[2] * 0.9)

        # 2. Novelty Score Calculation (Data-Driven)
        # Novelty is inversely proportional to the highest similarity score found in memory.
        if not memory_scores:
            novelty_score = 1.0
        else:
            # The highest similarity score tells us how "seen" this prompt is.
            max_similarity = max(memory_scores)
            # If max similarity is high (e.g., 0.9), novelty is low (0.1).
            novelty_score = 1.0 - max_similarity

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

    def get_long_term_memory(self, user_id: str, prompt: str) -> tuple[list[str], list[float]]:
        """
        Retrieves semantically relevant memory nodes using simulated vector search 
        and returns the keywords and similarity scores of the top nodes.
        """
        # Use the in-memory cache for performance.
        if user_id not in self.long_term_memory:
            # On first access, load from persistent storage into the cache.
            self.long_term_memory[user_id] = state_manager.get_memory_for_user(user_id)

        if not self.long_term_memory.get(user_id):
            return [], []

        # --- REALISTIC SEMANTIC SEARCH (SIMULATED) ---
        # 1. Generate a semantic vector for the prompt using the embedding model.
        prompt_vector = embedding_model.get_embedding(prompt)
        
        retrieval_scores = []

        for node in self.long_term_memory[user_id]:
            node_vector = np.array(node.core_intent_vector)
            # 2. Simulated Similarity (Cosine Similarity placeholder)
            norm_product = np.linalg.norm(prompt_vector) * np.linalg.norm(node_vector)
            similarity_score = np.dot(prompt_vector, node_vector) / norm_product if norm_product != 0 else 0

            # 3. Apply Weighting (Recency + Performance)
            weighted_score = similarity_score * self._calculate_weight(node)
            
            # Store the raw similarity for novelty calculation, and the weighted score for ranking.
            retrieval_scores.append({"weighted_score": weighted_score, "similarity": similarity_score, "node": node})

        # 4. Sort and Select Top N
        retrieval_scores.sort(key=lambda x: x["weighted_score"], reverse=True)
        top_results = retrieval_scores[:MEMORY_RETRIEVAL_LIMIT]

        # 5. Extract Keywords
        keywords_list = []
        similarity_scores = []
        for result in top_results:
            keywords_list.extend(result["node"].keywords)
            similarity_scores.append(result["similarity"])
            
        return list(set(keywords_list)), similarity_scores

    def _create_memory_node(self, cognitive_packet: CognitivePacket) -> MemoryNode:
        """Creates a MemoryNode from a CognitivePacket, calculating performance."""
        # 1. Calculate performance score based on WGPMHI audit results.
        # A simple metric: ratio of 'Pass' results to total tests run.
        pass_count = sum(1 for result in cognitive_packet.wgpmhi_results.values() if "Pass" in str(result))
        total_tests = len(cognitive_packet.wgpmhi_results)
        # Scale score between 0.5 (minimum viability) and 1.0 (perfect).
        performance_score = 0.5 + (pass_count / total_tests) * 0.5 if total_tests > 0 else 0.5

        # 2. Generate a core intent vector using the semantic embedding model.
        primary_intent = cognitive_packet.intent.get("primary", "")
        prompt_vector = embedding_model.get_embedding(primary_intent)

        # 3. Extract keywords.
        keywords = [tag['value'] for tag in StrategicHeuristics()._generate_tags(primary_intent)]

        # 4. Assemble and return the MemoryNode.
        return MemoryNode(
            node_id=f"mn-{uuid.uuid4()}",
            timestamp=cognitive_packet.timestamp,
            core_intent_vector=prompt_vector.tolist(), # type: ignore
            keywords=keywords,
            performance_score=round(performance_score, 4),
            packet_reference=cognitive_packet
        )

    def update_long_term_memory(self, user_id: str, cognitive_packet: Union[CognitivePacket, str]):
        """Creates a MemoryNode and adds it to the user's long-term memory."""
        if isinstance(cognitive_packet, CognitivePacket):
            # Ensure the user's memory cache is initialized.
            if user_id not in self.long_term_memory:
                self.long_term_memory[user_id] = state_manager.get_memory_for_user(user_id)
            # Append to the cache and then save the entire updated cache to persistent storage.
            new_node = self._create_memory_node(cognitive_packet)
            self.long_term_memory[user_id].append(new_node)
            state_manager.save_memory_for_user(user_id, self.long_term_memory[user_id])
        elif isinstance(cognitive_packet, str):
            # This path is for anti-fragility learning summaries.
            log.info(f"Anti-fragility learning for user '{user_id}': {cognitive_packet}")

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
        This method now also resolves identified ambiguities from the sense-making phase.
        """
        # --- Ambiguity Resolution ---
        ambiguity_analysis = blueprint.ambiguity_analysis
        
        # 1. Resolve Contradictions
        if "Prompt asks for both brevity and comprehensiveness." in ambiguity_analysis.get("contradictions", []):
            # Decision: Prioritize the explicit word limit as it's a harder constraint.
            # Add a warning to acknowledge the conflict.
            blueprint.constraints.append("CONSISTENCY_WARNING: Prompt contained conflicting requests for brevity and comprehensiveness. Prioritizing brevity.")

        # 2. Clarify Ambiguous Terms
        if "best" in ambiguity_analysis.get("ambiguous_terms", []):
            # Decision: Define "best" in a measurable way for this context.
            blueprint.constraints.append("CLARIFICATION(best=highest_performance_score)")

        # --- Consistency Check ---
        # Conflict between a comprehensive latent intent and a low word limit.
        is_comprehensive_intent = "framework" in blueprint.latent_intent or "comprehensive" in blueprint.latent_intent
        word_limit_constraint = next((c for c in blueprint.constraints if c.startswith("word_limit:")), None)

        if is_comprehensive_intent and word_limit_constraint:
            try:
                limit = int(word_limit_constraint.split(':')[1])
                if limit < 150:  # Arbitrary threshold for "too brief" for a framework
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
            "primary_intent": primary_intent, # @RISK: Raw prompt, may contain sensitive info.
            "latent_intent": latent_intent,
        }

    def _generate_tags(self, text: str) -> List[Dict[str, str]]:
        """Helper for tag generation. Can be expanded to include conversational context."""
        acronyms = {'ai', 'ml', 'gnn', 'ann'}
        words = {word for word in re.sub(r'[^\w\s]', '', text.lower()).split() if len(word) > 4 or word in acronyms}
        return [{"type": "keyword", "value": keyword} for keyword in sorted(list(words))]

    def generate_tags(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generates Math Language tags based on the context, influenced by long-term memory."""
        prompt = context.get("prompt", "")
        context_evaluation = context.get("context_evaluation", {"novelty_score": 0.0, "risk_score": 0.0})
        long_term_memory = context.get("long_term_memory", [])

        # --- CASSANDRA Heuristics: Start with strategic tags based on context evaluation ---
        strategic_tags = []
        novelty_score = context_evaluation.get("novelty_score", 0.0)
        risk_score = context_evaluation.get("risk_score", 0.0)

        # 1. RISK-ADJUSTED PLANNING
        if risk_score > 0.5:
            strategic_tags.append({"type": "RISK_LEVEL", "value": "HIGH"})
            strategic_tags.append({"type": "STRATEGIC_PROTOCOL", "value": "SAFETY_PRIORITY"})
            strategic_tags.append({"type": "ETHICAL_ALIGNMENT", "value": "ETHICAL_CONSULT"})

        # 2. NOVELTY-ADJUSTED PLANNING
        if novelty_score > 0.8:
            strategic_tags.append({"type": "CONTEXT_CONFIDENCE", "value": "LOW"})
            strategic_tags.append({"type": "RESOURCE_ALLOCATION", "value": "REQUIRE_EXTERNAL_DATA"})

        # --- CHRONOS Heuristics: Predictive Modeling Detection ---
        predictive_keywords = ["predict", "forecast", "trend", "time-series", "sequential data"]
        if any(keyword in prompt.lower() for keyword in predictive_keywords):
            if "PREDICTIVE_MODEL: REQUIRED" not in {t['value'] for t in strategic_tags}:
                strategic_tags.append({"type": "MODEL_PROTOCOL", "value": "PREDICTIVE_MODEL: REQUIRED"})

        # --- Keyword Extraction (from prompt and memory) ---
        # This is a more efficient approach that avoids redundant function calls and set operations.
        existing_tag_values = {t['value'] for t in strategic_tags}
        acronyms = {'ai', 'ml', 'gnn', 'ann'}
        
        # Combine text from prompt and all memory keywords into a single string for processing.
        combined_text = prompt + " " + " ".join(long_term_memory)
        
        words = {word for word in re.sub(r'[^\w\s]', '', combined_text.lower()).split() if len(word) > 4 or word in acronyms}
        keyword_tags = [{"type": "keyword", "value": keyword} for keyword in sorted(list(words)) if keyword not in existing_tag_values]

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
            tags=[{"type": t["type"], "value": t["value"]} for t in self.strategic_heuristics.generate_tags(context)], # Use full context for tags
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
            external_data=context.get("external_data"),
        )

        # Run the final integrity check on the assembled blueprint.
        blueprint = self.strategic_heuristics.verify_integrity(blueprint)

        return blueprint

    def refine_blueprint(self, blueprint: Blueprint, audit_failures: List[str]) -> Blueprint:
        """Public method to access the correction plan generation."""
        corrected_blueprint = self.strategic_heuristics._generate_correction_plan(blueprint, audit_failures)
        return corrected_blueprint

noesis_triad = NoesisTriad()