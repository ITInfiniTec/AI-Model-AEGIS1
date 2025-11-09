# wgpmhi.py

import re
from data_structures import Blueprint, UserProfile, CognitivePacket, MemoryNode, ExecutionPlan
import math
from cognitive_fallacy_library import cognitive_fallacy_library
import inspect
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from cmep import cmep
from noesis_triad import NoesisTriad
from logger import log
from config_loader import config_loader
from state_manager import state_manager
from config import MEMORY_DECAY_TAU, MEMORY_RETRIEVAL_LIMIT, set_memory_decay_tau, set_memory_retrieval_limit

class WadeGeminiProtocol:
    """
    The Wade-Gemini Protocol for Meta-Human Intelligence (WGPMHI).
    Serves as the integrated benchmark and audit suite for the AEGIS cognitive engine.
    """
    def __init__(self):
        # Project NEMESIS: Register self-healing strategies in a scalable way.
        # This replaces a rigid if/elif/else structure with a flexible dictionary.
        self._healing_strategies = {
            "memory_retrieval_weight_check": self._heal_memory_retrieval
        }
        # Load hallucination audit configuration
        hallucination_config = config_loader.get_hallucination_audit_config()
        self.topic_keywords = hallucination_config.get("topic_keywords", {})
        self.hallucination_stop_words = set(hallucination_config.get("stop_words", []))
        # Load WGPMHI audit configuration
        wgpmhi_config = config_loader.get_wgpmhi_audit_config()
        self.historical_risk_penalty = wgpmhi_config.get("historical_risk_penalty", 0.1)
        self.healing_tau_multiplier = wgpmhi_config.get("healing_tau_multiplier", 1.2)
        self.safety_tag_threshold = wgpmhi_config.get("safety_tag_threshold", 0.5)
        self.sentinel_persona_threshold = wgpmhi_config.get("sentinel_persona_threshold", 0.7)
        self.high_concept_threshold = wgpmhi_config.get("high_conceptual_integration_threshold", 3)
        self.basic_concept_threshold = wgpmhi_config.get("basic_conceptual_integration_threshold", 1)
        # This is now defined in the config but was missing from the loader.
        self.novelty_response_threshold = wgpmhi_config.get("novelty_response_threshold", 0.8)
        # Load STG dependency rules for validation
        stg_rules = config_loader.get_stg_dependency_rules()
        self.stg_op_types = stg_rules.get("operation_types", {})
        self.stg_dependencies = stg_rules.get("dependencies", {})
        # Load ambiguity rules for validation
        self.ambiguity_rules = config_loader.get_ambiguity_rules()
        # Load compiler rules for validation
        self.compiler_rules = config_loader.get_compiler_rules()
        # Load outcome rules for validation
        self.outcome_rules = config_loader.get_outcome_rules()
        # Load tag generation rules for validation
        tag_gen_config = config_loader.get_tag_generation_config()
        self.predictive_keywords = tag_gen_config.get("predictive_keywords", [])
        # Load output formatting strings for validation
        output_formatting_config = config_loader.get_output_formatting_config()
        self.low_confidence_note_prefix = output_formatting_config.get("low_confidence_note", "").split(':')[0] + ':'

    def _heal_memory_retrieval(self) -> str:
        """Anti-fragility action for memory retrieval failures."""
        old_tau = MEMORY_DECAY_TAU
        # Increase TAU to make memories decay slower, using a configurable multiplier.
        new_tau = old_tau * self.healing_tau_multiplier
        set_memory_decay_tau(new_tau)
        return f"PROTOCOL ACTIVE: Adjusted MEMORY_DECAY_TAU from {old_tau:.2f} to {new_tau:.2f} due to Memory Retrieval Failure."

    def _trigger_antifragility_protocol(self, test_name: str, failure_reason: str, user_id: str, noesis_triad: NoesisTriad) -> Optional[str]:
        """
        Project NEMESIS: Adjusts system parameters based on critical test failures,
        making the system anti-fragile by learning from stress.
        This now uses a dictionary lookup for scalability.
        """
        learning_summary = f"ANTI-FRAGILITY_LEARNING: Test '{test_name}' failed. Reason: '{failure_reason}'."
        # Log the event using the centralized logger. This is the correct behavior.
        log.warning(f"{learning_summary} - UserID: {user_id}")

        # Execute the registered healing strategy if one exists for the failed test.
        healing_function = self._healing_strategies.get(test_name)
        return healing_function() if healing_function else None

    def run_pre_compilation_audit(self, blueprint: Blueprint, execution_plan: ExecutionPlan) -> List[str]:
        """
        Analyzes the blueprint and structured ExecutionPlan for internal consistency.
        Returns a list of failure descriptions.
        """
        failures = []
        plan_operations = {details['operation'] for details in execution_plan.stg.values()}

        # --- Resource Allocation Check ---
        # Verifies that if the blueprint requires external data, the plan includes an operation to fetch it.
        requires_external_data = any(tag.type == "RESOURCE_ALLOCATION" and tag.value == "REQUIRE_EXTERNAL_DATA" for tag in blueprint.tags)
        has_fetch_operation = any(op.startswith("OP_FETCH_KNOWLEDGE") or op.startswith("OP_FETCH_TIME_SERIES_DATA") for op in plan_operations)

        if requires_external_data and not has_fetch_operation:
            failures.append("Resource_Allocation: Blueprint required external data, but no fetch operation was included in the execution plan.")

        # This check is now handled by the main audit suite's _check_latent_intent_actuation
        # It remains here as an example of a pre-compilation check.

        # Add other pre-compilation checks here as needed.

        return failures

    # --- Individual Test Implementations ---

    def _check_ethical_compass(self, blueprint: Blueprint, user_profile: UserProfile) -> str:
        """
        Checks if the blueprint correctly flags ethical concerns, including both
        hard red-lines and softer user-value alignments for controversial topics.
        """
        # Replicate the logic from NoesisTriad to determine if a flag was expected.
        prompt = blueprint.primary_intent
        should_have_red_line_flag = cmep.check_red_lines(prompt)
        should_have_alignment_flag = cmep.align_with_user_values(prompt, user_profile) != prompt
        should_flag = should_have_red_line_flag or should_have_alignment_flag

        was_flagged = "violation detected" in blueprint.ethical_considerations
        return "Pass" if should_flag == was_flagged else f"Fail: Discrepancy in ethical check. Expected flag: {should_flag}, but was_flagged: {was_flagged}."

    def _check_cognitive_logic(self, output: str) -> str:
        return "Fail: Output does not align with original intent." if "[AUDIT_FAIL]" in output else "Pass"

    def _check_strategic_reasoning(self, blueprint: Blueprint, execution_plan: ExecutionPlan) -> str:
        """
        Checks if constraints from the blueprint are correctly translated into the execution plan.
        1. Verifies that the raw constraint list is passed through.
        2. Verifies that specific constraints (like AUDIENCE and FORMAT) are correctly parsed.
        """
        # 1. Check if the raw list of constraints is preserved.
        if not all(c in execution_plan.constraints for c in blueprint.constraints):
            return f"Fail: Not all blueprint constraints {blueprint.constraints} were found in the compiled plan {execution_plan.constraints}."

        # 2. Check if specific constraints were correctly parsed into their dedicated fields.
        expected_audience = next((c.split('(')[1].strip(')') for c in blueprint.constraints if c.startswith("AUDIENCE")), "GENERAL_USER")
        expected_format = next((c.split('(')[1].strip(')') for c in blueprint.constraints if c.startswith("FORMAT")), "TEXT_BLOCK")

        if expected_audience != execution_plan.target_audience: return f"Fail: Blueprint audience '{expected_audience}' was not correctly parsed into plan's target_audience '{execution_plan.target_audience}'."
        if expected_format != execution_plan.target_format: return f"Fail: Blueprint format '{expected_format}' was not correctly parsed into plan's target_format '{execution_plan.target_format}'."
        return "Pass"

    def _check_philosophical_synthesis(self, blueprint: Blueprint) -> str:
        if len(blueprint.tags) >= self.high_concept_threshold:
            return "Pass: High Conceptual Integration"
        return "Pass: Basic Conceptual Presence" if len(blueprint.tags) >= self.basic_concept_threshold else "Pass: Low Conceptual Integration"

    def _check_creative_conundrum(self, blueprint: Blueprint, execution_plan: ExecutionPlan) -> str:
        """
        Checks if creative intent is correctly identified and translated into the
        appropriate creative operation in the execution plan, based on compiler rules.
        """
        creative_rules = [rule for rule in self.compiler_rules if rule.get("operation", "").startswith("OP_CREATIVE_")]
        prompt_tags = {tag.value for tag in blueprint.tags}
        plan_operations = {details['operation'] for details in execution_plan.stg.values()}
        
        intent_detected = False
        for rule in creative_rules:
            trigger_keywords = rule.get("conditions", {}).get("tags_include_any", [])
            if any(kw in prompt_tags for kw in trigger_keywords):
                intent_detected = True
                if rule["operation"] not in plan_operations:
                    return f"Fail: Creative intent for '{rule['rule_name']}' detected, but operation '{rule['operation']}' was not in the plan."

        if not intent_detected:
            return "N/A"
        return "Pass"

    def _check_ethical_conflict_resolution(self, blueprint: Blueprint) -> str:
        """
        Checks if a user-value conflict was detected and resolved correctly.
        This test now uses the `original_intent` for a precise, non-circular check.
        """
        # Compare the original, unaltered prompt with the final primary intent.
        was_rewritten = blueprint.original_intent != blueprint.primary_intent
        was_flagged = "user-value alignment detected" in blueprint.ethical_considerations

        if not was_rewritten and not was_flagged:
            return "N/A"
        
        if was_rewritten and not was_flagged:
            return "Fail: Prompt was rewritten for ethical alignment, but the corresponding flag was not set in ethical_considerations."
        if not was_rewritten and was_flagged:
            return "Fail: An ethical alignment flag was set, but the prompt was not rewritten."
        return "Pass"

    def _check_self_correction_audit(self, blueprint: Blueprint, execution_plan: ExecutionPlan) -> str:
        """Checks if the fallacies detected in the blueprint are correctly reported in the execution plan."""
        blueprint_fallacies = set(blueprint.fallacies)
        plan_warnings = set(execution_plan.fallacy_warnings)

        if blueprint_fallacies == plan_warnings:
            return "Pass"

        missed_in_plan = blueprint_fallacies - plan_warnings
        extra_in_plan = plan_warnings - blueprint_fallacies

        error_messages = []
        if missed_in_plan: error_messages.append(f"Missed warnings for: {', '.join(missed_in_plan)}")
        if extra_in_plan: error_messages.append(f"Extraneous warnings for: {', '.join(extra_in_plan)}")
        return f"Fail: Discrepancy in fallacy reporting. {'; '.join(error_messages)}."

    def _check_memory_continuity(self, user_profile: UserProfile, blueprint: Blueprint, noesis_triad: NoesisTriad) -> str:
        """Checks if keywords from past interactions are present in the current blueprint's tags."""
        # This test now uses the Noesis Triad's own memory retrieval logic to ensure
        # it's testing the actual context that was built for the blueprint.
        # It's more efficient and a more accurate test of the system's behavior.
        relevant_memory_keywords, _ = noesis_triad.context_synthesizer.get_long_term_memory(user_profile.user_id, blueprint.primary_intent)

        if not relevant_memory_keywords:
            return "N/A (No prior memory to test)"

        # The blueprint tags should include keywords from relevant past interactions.
        blueprint_tags = {tag.value for tag in blueprint.tags}
        continuity_found = any(kw in blueprint_tags for kw in relevant_memory_keywords)

        return "Pass" if continuity_found else "Fail: No keywords from relevant memories were found in the current blueprint's tags."

    def _check_hallucination_ratio(self, blueprint: Blueprint, execution_plan: ExecutionPlan, output: str) -> str:
        """
        Checks if the output prose aligns with the knowledge used.
        1. If external data was fetched, it verifies the prose reflects that data.
        2. Otherwise, it checks for internal consistency with planned knowledge topics.
        """
        prose_match = re.search(r"--- SIMULATED PROSE OUTPUT ---\n(.*)", output, re.DOTALL)
        if not prose_match:
            return "N/A"
        prose_output = prose_match.group(1).lower()

        # Case 1: Verify against external data (higher priority)
        if blueprint.external_data:
            # Dynamically extract keywords from the external data source to avoid hardcoding.
            # This makes the test robust and data-driven.
            source_text = blueprint.external_data.replace("SIMULATED_EXTERNAL_DATA:", "").lower()
            
            # Extract significant words from the source text.
            source_words = set(re.sub(r'[^\w\s]', '', source_text).split())
            external_keywords = [word for word in source_words if word and word not in self.hallucination_stop_words]

            if not all(kw in prose_output for kw in external_keywords):
                return f"Fail: Prose does not reflect key concepts from fetched external data. Missing: {[kw for kw in external_keywords if kw not in prose_output]}."

            return "Pass"

        # Case 2: Verify against internal knowledge topics if no external data was used
        topics = [op.split("'")[1] for details in execution_plan.stg.values() if (op := details.get('operation', '')).startswith("OP_FETCH_KNOWLEDGE")]
        if not topics: return "Pass (No knowledge topics to verify)" # type: ignore
        
        missing_topics = [topic for topic in topics if not any(kw in prose_output for kw in self.topic_keywords.get(topic, [topic]))]
        return "Pass" if not missing_topics else f"Fail: Prose seems to miss topics from execution plan: {', '.join(missing_topics)}"

    def _check_outcome_alignment(self, blueprint: Blueprint) -> str:
        """
        Checks if the expected outcome was correctly parsed from the prompt
        based on the configured outcome rules.
        """
        # Re-run the extraction logic to find what the expected outcome *should* have been.
        expected_outcome_from_prompt = ""
        for pattern in self.outcome_rules:
            match = re.search(pattern, blueprint.primary_intent.lower())
            if match:
                expected_outcome_from_prompt = match.group(1).strip()
                break # Found the first match, which is what the generator does.
        
        return "Pass" if blueprint.expected_outcome == expected_outcome_from_prompt else f"Fail: Blueprint's expected outcome '{blueprint.expected_outcome}' does not match the expected value '{expected_outcome_from_prompt}' parsed from the prompt."

    def _check_latent_intent_actuation(self, blueprint: Blueprint, execution_plan: ExecutionPlan) -> str:
        """
        Checks if latent intent is correctly translated into operations based on compiler rules.
        This is now a dynamic check against the configuration, not a hardcoded value.
        """
        latent_intent_rules = [rule for rule in self.compiler_rules if "latent_intent_is" in rule.get("conditions", {})]
        if not latent_intent_rules:
            return "N/A"

        plan_operations = {details['operation'] for details in execution_plan.stg.values()}
        for rule in latent_intent_rules:
            if blueprint.latent_intent == rule["conditions"]["latent_intent_is"]:
                if rule["operation"] not in plan_operations:
                    return f"Fail: Latent intent for '{rule['rule_name']}' detected, but operation '{rule['operation']}' was not in the plan."
        return "Pass"

    def _check_data_integrity(self, user_profile: UserProfile) -> str:
        """
        Checks the integrity of the UserProfile, ensuring all numeric values
        are within their expected ranges and that other fields are well-formed.
        """
        # 1. Validate numeric values in the 'values' dictionary.
        for key, value in user_profile.values.items():
            if isinstance(value, float): # Check all float values for range validity
                if not 0.0 <= value <= 1.0:
                    return f"Fail: UserProfile value '{key}' ({value}) is outside the validated range [0.0, 1.0]."
        
        # 2. Validate the structure and content of the 'passions' list.
        if not isinstance(user_profile.passions, list):
            return f"Fail: UserProfile 'passions' field is not a list (type: {type(user_profile.passions).__name__})."
        
        for passion in user_profile.passions:
            if not isinstance(passion, str) or not passion:
                return f"Fail: UserProfile 'passions' list contains an invalid or empty item: '{passion}'."

        return "Pass"

    def _check_output_constraint_alignment(self, blueprint: Blueprint, execution_plan: ExecutionPlan, output: str) -> str:
        """Checks if the final execution plan and output string respect blueprint constraints."""
        blueprint_audience = next((c.split('(')[1].strip(')') for c in blueprint.constraints if c.startswith("AUDIENCE")), "GENERAL_USER")
        blueprint_format = next((c.split('(')[1].strip(')') for c in blueprint.constraints if c.startswith("FORMAT")), "TEXT_BLOCK")

        # 1. Verify parsed constraints in the execution plan.
        if blueprint_audience != execution_plan.target_audience: return f"Fail: Blueprint audience '{blueprint_audience}' != plan audience '{execution_plan.target_audience}'."
        if blueprint_format != execution_plan.target_format: return f"Fail: Blueprint format '{blueprint_format}' != plan format '{execution_plan.target_format}'."

        # 2. Verify low-confidence note is present in the output.
        confidence_tag = next((tag for tag in blueprint.tags if tag.type == "CONTEXT_CONFIDENCE"), None)
        if confidence_tag and confidence_tag.value == "LOW" and self.low_confidence_note_prefix not in output:
            return f"Fail: Low context confidence was detected, but the '{self.low_confidence_note_prefix}' was not included in the output."

        # 3. Verify that the format constraint was applied in the final output.
        if blueprint_format != "TEXT_BLOCK":
            if f"SIMULATED_FORMAT_APPLIED({blueprint_format})" not in output:
                return f"Fail: Format constraint '{blueprint_format}' was not applied in the final output."

        # 4. Verify word limit constraint was enforced in the final output.
        limit = execution_plan.word_limit
        if limit is not None:
            word_count = len(output.split())
            if word_count > limit + 1: # Allow one extra word for the ellipsis.
                return f"Fail: Word count ({word_count}) exceeded the specified limit of {limit}."
        return "Pass"

    def _check_risk_adjusted_planning(self, blueprint: Blueprint, noesis_triad: NoesisTriad) -> str:
        """
        Checks if the system correctly identifies risky prompts and adjusts its planning.
        1. Verifies consistency of the risk score in the blueprint.
        2. Checks if high risk correctly triggers required safety tags.
        3. Checks if high risk correctly triggers a more cautious persona.
        """
        # 1. Re-calculate risk to ensure consistency.
        temp_context = noesis_triad.context_synthesizer.build_context(blueprint.user_id, blueprint.primary_intent)
        context_evaluation = noesis_triad.context_synthesizer.evaluate_context_risk(temp_context)
        recalculated_risk = context_evaluation.get("risk_score", -1.0)
        
        # --- Historical Audit Analysis ---
        # Retrieve similar memory nodes to see if this topic has been problematic in the past.
        similar_nodes, _ = noesis_triad.context_synthesizer.get_long_term_memory(blueprint.user_id, blueprint.primary_intent)
        historical_risk_factor = 0.0
        
        # Check the audit history of similar past interactions.
        for node in similar_nodes:
            past_audits = node.packet_reference.wgpmhi_results
            # If similar prompts led to logic or hallucination failures, increase risk.
            if "Fail" in past_audits.get("cognitive_logic", "") or "Fail" in past_audits.get("hallucination_ratio", ""):
                historical_risk_factor += self.historical_risk_penalty # Add a risk penalty for each past failure.
        
        # Boost the recalculated risk with the historical factor.
        final_risk = min(1.0, recalculated_risk + historical_risk_factor)

        # The blueprint's risk score should reflect this combined risk.
        if not math.isclose(blueprint.risk_score, final_risk, rel_tol=1e-2):
            return f"Fail: Blueprint risk score ({blueprint.risk_score:.4f}) is inconsistent with historically-adjusted score ({final_risk:.4f})."

        # 2. Check if high risk correctly triggers required safety tags.
        blueprint_tag_values = {t.value for t in blueprint.tags}
        if final_risk > self.safety_tag_threshold and ("SAFETY_PRIORITY" not in blueprint_tag_values or "ETHICAL_CONSULT" not in blueprint_tag_values):
            return "Fail: High risk score (>0.5) but missing required safety tags."

        # 3. Check if high risk correctly triggers the Sentinel persona.
        if final_risk > self.sentinel_persona_threshold and blueprint.persona != "The_Sentinel":
            return f"Fail: High risk score ({final_risk:.4f}) did not trigger the 'The_Sentinel' persona."

        return "Pass"

    def _check_novelty_awareness(self, blueprint: Blueprint, noesis_triad: NoesisTriad) -> str:
        """
        Checks if the system correctly identifies novel prompts and adjusts its confidence.
        1. Verifies consistency of the novelty score.
        2. Checks if high novelty triggers a low confidence tag.
        3. Checks if high novelty triggers a requirement for external data.
        """
        # 1. Re-calculate novelty to ensure consistency.
        temp_context = noesis_triad.context_synthesizer.build_context(blueprint.user_id, blueprint.primary_intent)
        context_evaluation = noesis_triad.context_synthesizer.evaluate_context_risk(temp_context)
        recalculated_novelty = context_evaluation.get("novelty_score", -1.0)

        if not math.isclose(blueprint.novelty_score, recalculated_novelty, rel_tol=1e-2):
            return f"Fail: Blueprint novelty score ({blueprint.novelty_score:.4f}) is inconsistent with recalculated score ({recalculated_novelty:.4f})."

        # 2. Check if high novelty correctly triggers low confidence.
        has_low_confidence_tag = any(tag.type == "CONTEXT_CONFIDENCE" and tag.value == "LOW" for tag in blueprint.tags)
        if blueprint.novelty_score > self.novelty_response_threshold and not has_low_confidence_tag:
            return f"Fail: High novelty score ({blueprint.novelty_score:.4f}) did not trigger a 'CONTEXT_CONFIDENCE: LOW' tag."

        # 3. Check if high novelty correctly triggers a requirement for external data.
        requires_external_data = any(tag.type == "RESOURCE_ALLOCATION" and tag.value == "REQUIRE_EXTERNAL_DATA" for tag in blueprint.tags)
        if blueprint.novelty_score > self.novelty_response_threshold and not requires_external_data:
            return f"Fail: High novelty score ({blueprint.novelty_score:.4f}) did not trigger a 'RESOURCE_ALLOCATION: REQUIRE_EXTERNAL_DATA' tag."

        return "Pass"

    def _check_ambiguity_resolution(self, blueprint: Blueprint) -> str:
        """
        Checks if the system correctly resolves identified ambiguities based on the configured rules.
        """
        ambiguity_analysis = blueprint.ambiguity_analysis
        contradictions = ambiguity_analysis.get("contradictions", [])
        ambiguous_terms = ambiguity_analysis.get("ambiguous_terms", [])
        blueprint_constraints = set(blueprint.constraints)

        if not contradictions and not ambiguous_terms:
            return "N/A (No ambiguities detected in blueprint)"

        # Validate that for each detected ambiguity, the correct resolution constraint was applied.
        for rule in self.ambiguity_rules.get("contradictions", []):
            if f"Contradiction between: {', '.join(rule['terms'])}" in contradictions:
                if rule["resolution"] not in blueprint_constraints:
                    return f"Fail: Contradiction for '{rule['terms']}' was detected but resolution '{rule['resolution']}' was not applied."

        for rule in self.ambiguity_rules.get("ambiguous_terms", []):
            if rule["term"] in ambiguous_terms:
                if rule["resolution"] not in blueprint_constraints:
                    return f"Fail: Ambiguous term '{rule['term']}' was detected but resolution '{rule['resolution']}' was not applied."

        return "Pass"

    def _check_memory_node_creation(self, cognitive_packet: Optional[CognitivePacket], noesis_triad: NoesisTriad) -> str:
        """Checks if a MemoryNode is created with valid, non-empty data."""
        if not cognitive_packet:
            return "Skipped (Cognitive Packet not yet available)"
        try:
            test_node = noesis_triad.context_synthesizer._create_memory_node(cognitive_packet)
            
            if not isinstance(test_node, MemoryNode): return "Fail: _create_memory_node did not return a MemoryNode object."
            if not (0.5 <= test_node.performance_score <= 1.0): return f"Fail: Performance score ({test_node.performance_score}) is outside the valid range [0.5, 1.0]."
            
            # Check that the core vector data was generated correctly.
            if not test_node.core_intent_vector or not isinstance(test_node.core_intent_vector[0], float):
                return "Fail: Core intent vector is empty or invalid."

            # Verify that the keywords stored in the node are correct by re-generating them.
            primary_intent = cognitive_packet.intent.get("primary", "")
            expected_keywords = {tag['value'] for tag in noesis_triad.strategic_heuristics._generate_tags(primary_intent)}
            actual_keywords = set(test_node.keywords)

            if expected_keywords != actual_keywords:
                return f"Fail: Keywords in memory node do not match expected keywords. Missing: {expected_keywords - actual_keywords}, Extra: {actual_keywords - expected_keywords}."

            return "Pass"
        except Exception as e:
            return f"Fail: Exception during node creation simulation: {str(e)}"

    def _check_memory_retrieval_weight(self, cognitive_packet: CognitivePacket, noesis_triad: NoesisTriad) -> str:
        """
        Checks that the memory weighting function correctly prioritizes nodes
        based on recency and performance score, with isolated checks for each factor.
        """
        if not cognitive_packet:
            return "Skipped (Cognitive Packet not yet available)"
        try:
            now = datetime.now()
            # 1. Isolate Recency: Same performance, different timestamps.
            recent_node = MemoryNode(node_id="TEST_RECENT", timestamp=now - timedelta(days=1), core_intent_vector=[1.0], keywords=["test"], performance_score=0.8, packet_reference=cognitive_packet)
            old_node = MemoryNode(node_id="TEST_OLD", timestamp=now - timedelta(days=30), core_intent_vector=[1.0], keywords=["test"], performance_score=0.8, packet_reference=cognitive_packet)
            if noesis_triad.context_synthesizer._calculate_weight(recent_node) <= noesis_triad.context_synthesizer._calculate_weight(old_node):
                return "Fail: Weight calculation is not correctly prioritizing recency."

            # 2. Isolate Performance: Same timestamp, different performance.
            high_perf_node = MemoryNode(node_id="TEST_HIGH_PERF", timestamp=now, core_intent_vector=[1.0], keywords=["test"], performance_score=1.0, packet_reference=cognitive_packet)
            low_perf_node = MemoryNode(node_id="TEST_LOW_PERF", timestamp=now, core_intent_vector=[1.0], keywords=["test"], performance_score=0.5, packet_reference=cognitive_packet)
            if noesis_triad.context_synthesizer._calculate_weight(high_perf_node) <= noesis_triad.context_synthesizer._calculate_weight(low_perf_node):
                return "Fail: Weight calculation is not correctly prioritizing performance score."

            # 3. Combined Check (original test)
            if noesis_triad.context_synthesizer._calculate_weight(high_perf_node) <= noesis_triad.context_synthesizer._calculate_weight(old_node):
                return "Fail: Weight calculation is not correctly prioritizing combined recency and performance."
            return "Pass"
        except Exception as e:
            return f"Fail: Exception during weight check: {str(e)}"

    def _get_stg_op_type(self, operation: str) -> str | None:
        """Helper to find the type of a given operation string for STG validation."""
        for op_type, op_list in self.stg_op_types.items():
            if any(op_name in operation for op_name in op_list):
                return op_type
        return None

    def _check_stg_dependency(self, execution_plan: ExecutionPlan) -> str:
        """
        Validates the STG against the configured dependency rules.
        Ensures that tasks correctly depend on their prerequisites.
        """
        stg = execution_plan.stg
        if not stg:
            return "N/A (No STG to validate)"

        tasks_by_type: Dict[str, List[str]] = {op_type: [] for op_type in self.stg_op_types}
        for task_id, details in stg.items():
            op_type = self._get_stg_op_type(details["operation"])
            if op_type:
                tasks_by_type[op_type].append(task_id)

        for task_id, details in stg.items():
            op_type = self._get_stg_op_type(details["operation"])
            if op_type and op_type in self.stg_dependencies:
                for required_dep_type in self.stg_dependencies[op_type]:
                    required_tasks = tasks_by_type.get(required_dep_type, [])
                    if not all(req_task in details["depends_on"] for req_task in required_tasks):
                        return f"Fail: Task {task_id} ({op_type}) is missing a dependency on {required_dep_type} tasks."
        return "Pass"

    def _check_predictive_workflow(self, blueprint: Blueprint, execution_plan: ExecutionPlan, output: str) -> str:
        """
        Checks the entire predictive workflow.
        1. Verifies that predictive keywords in the prompt correctly generate the required tag.
        2. Verifies the STG dependency chain for predictive tasks.
        3. Verifies that the final forecast is included in the output.
        """
        has_predictive_keyword = any(kw in blueprint.primary_intent.lower() for kw in self.predictive_keywords)
        has_predictive_tag = any(tag.value == "PREDICTIVE_MODEL: REQUIRED" for tag in blueprint.tags)

        if not has_predictive_keyword and not has_predictive_tag:
            return "N/A"

        if has_predictive_keyword and not has_predictive_tag:
            return "Fail: Prompt contained predictive keywords, but the 'PREDICTIVE_MODEL: REQUIRED' tag was not generated."

        stg = execution_plan.stg
        try:
            fetch_task_id = next(tid for tid, details in stg.items() if details['operation'] == 'OP_FETCH_TIME_SERIES_DATA')
            analyze_task_id, analyze_task_details = next((tid, details) for tid, details in stg.items() if details['operation'].startswith('OP_ANALYZE_SERIES'))
            forecast_task_id, forecast_task_details = next((tid, details) for tid, details in stg.items() if details['operation'] == 'OP_GENERATE_FORECAST')
        except StopIteration:
            return "Fail: One or more required predictive operations are missing from the STG."

        # Verify that the dependency chain is correct: Forecast -> Analyze -> Fetch
        if fetch_task_id not in analyze_task_details.get('depends_on', []):
            return f"Fail: Analyze task {analyze_task_id} does not depend on fetch task {fetch_task_id}."
        if analyze_task_id not in forecast_task_details.get('depends_on', []):
            return "Fail: Predictive task dependencies are incorrect."

        # Verify that if a forecast was generated, it was included in the final output.
        forecast_result = execution_plan.simulated_forecast_result
        if forecast_result and "No Time Series Data Analyzed" not in forecast_result:
            if forecast_result not in output:
                return "Fail: A forecast was generated but not included in the final output."
        return "Pass"

    def run_tests(self, user_profile: UserProfile, blueprint: Blueprint, execution_plan: ExecutionPlan, output: str, noesis_triad: NoesisTriad, cognitive_packet: Optional[CognitivePacket]):
        """
        Runs the full suite of WGPMHI tests using automated discovery.
        Any method prefixed with '_check_' is automatically discovered and executed.
        This is a professional practice that reduces boilerplate and prevents tests
        from being accidentally omitted from the run.
        """
        results = {}
        # A mapping of argument names to their values for dynamic test execution.
        # This makes the test runner flexible to methods with different signatures.
        arg_map = {
            "user_profile": user_profile,
            "blueprint": blueprint,
            "execution_plan": execution_plan,
            "output": output,
            "noesis_triad": noesis_triad,
            "cognitive_packet": cognitive_packet,
        }

        # Automated test discovery
        for attr_name in dir(self):
            if not attr_name.startswith("_check_"):
                continue

            test_func = getattr(self, attr_name)
            test_name = attr_name.replace("_check_", "")

            # Intelligently build the argument list for the test function.
            func_params = inspect.signature(test_func).parameters
            args_to_pass = {param: arg_map[param] for param in func_params if param in arg_map}

            try:
                results[test_name] = test_func(**args_to_pass)
            except Exception as e:
                results[test_name] = f"ERROR: {type(e).__name__} - {e}"

        protocol_actions = []
        # Trigger Anti-Fragility Protocol for any failures
        for test, result in results.items():
            if "Fail" in str(result):
                action_taken = self._trigger_antifragility_protocol(test, result, user_profile.user_id, noesis_triad)
                if action_taken:
                    protocol_actions.append(action_taken)
        results["anti_fragility_protocol_status"] = protocol_actions if protocol_actions else "PROTOCOL INACTIVE"

        # --- Final System Stability Check ---
        # This provides a single, top-level indicator of the system's health for this cycle.
        has_critical_failure = any("Fail" in str(res) or "ERROR" in str(res) for res in results.values())
        results["system_stability"] = "Fail" if has_critical_failure else "Pass"

        return results

wgpmhi = WadeGeminiProtocol()