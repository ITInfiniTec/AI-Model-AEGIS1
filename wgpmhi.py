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

    def _heal_memory_retrieval(self) -> str:
        """Anti-fragility action for memory retrieval failures."""
        old_tau = MEMORY_DECAY_TAU
        new_tau = old_tau * 1.2  # Increase TAU by 20% to make memories decay slower.
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

        # Run a subset of tests relevant to the execution plan's logic.
        # Latent Intent Actuation Check
        if "strategic framework" in blueprint.latent_intent:
            if "OP_GENERATE_STRATEGIC_FRAMEWORK" not in plan_operations:
                failures.append("Latent_Intent_Actuation: Latent intent for a framework was detected, but the corresponding operation was not executed.")

        # Add other pre-compilation checks here as needed.

        return failures

    # --- Individual Test Implementations ---

    def _check_ethical_compass(self, blueprint: Blueprint) -> str:
        should_flag = cmep.check_red_lines(blueprint.primary_intent)
        was_flagged = "violation detected" in blueprint.ethical_considerations
        return "Pass" if should_flag == was_flagged else f"Fail: Discrepancy in ethical check. Expected flag: {should_flag}, but was_flagged: {was_flagged}."

    def _check_cognitive_logic(self, output: str) -> str:
        return "Fail: Output does not align with original intent." if "[AUDIT_FAIL]" in output else "Pass"

    def _check_strategic_reasoning(self, blueprint: Blueprint, execution_plan: ExecutionPlan) -> str:
        """Checks if all constraints from the blueprint made it into the execution plan."""
        if not all(c in execution_plan.constraints for c in blueprint.constraints):
            return f"Fail: Not all blueprint constraints {blueprint.constraints} were found in the compiled plan {execution_plan.constraints}."
        return "Pass"

    def _check_philosophical_synthesis(self, blueprint: Blueprint) -> str:
        if len(blueprint.tags) >= 3:
            return "Pass: High Conceptual Integration"
        return "Pass: Basic Conceptual Presence" if len(blueprint.tags) >= 1 else "Pass: Low Conceptual Integration"

    def _check_creative_conundrum(self, blueprint: Blueprint, execution_plan: ExecutionPlan) -> str:
        creative_keywords = ['poem', 'story', 'imagine', 'create']
        if not any(keyword in blueprint.primary_intent.lower() for keyword in creative_keywords):
            return "N/A"
        plan_operations = {details['operation'] for details in execution_plan.stg.values()}
        return "Pass" if "OP_CREATIVE_WRITING" in plan_operations else "Fail: Creative intent detected but not processed."

    def _check_ethical_conflict_resolution(self, blueprint: Blueprint) -> str:
        if "user-value alignment detected" not in blueprint.ethical_considerations:
            return "N/A"
        return "Pass" if "Based on your preferences, this topic will be avoided" in blueprint.primary_intent else "Fail: Conflict detected but not resolved correctly."

    def _check_self_correction_audit(self, blueprint: Blueprint, execution_plan: ExecutionPlan) -> str:
        # This test should check the plan, not the final output string.
        if (blueprint.fallacies and execution_plan.fallacy_warnings) or (not blueprint.fallacies and not execution_plan.fallacy_warnings):
            return "Pass"
        return f"Fail: Discrepancy between blueprint fallacies ({blueprint.fallacies}) and plan warnings ({execution_plan.fallacy_warnings})."

    def _check_memory_continuity(self, user_profile: UserProfile, blueprint: Blueprint, noesis_triad: NoesisTriad) -> str:
        """Checks if keywords from past interactions are present in the current blueprint's tags."""
        # This test must query the StateManager directly to get the true state of memory,
        # rather than relying on the transient context passed to the NoesisTriad.
        memory_nodes = state_manager.get_memory_for_user(user_profile.user_id)
        if not memory_nodes:
            return "N/A (No prior memory to test)"

        memory_keywords = {kw for node in memory_nodes for kw in node.keywords}
        prompt_keywords = {tag['value'] for tag in noesis_triad.strategic_heuristics._generate_tags(blueprint.primary_intent)}
        continuity_candidates = memory_keywords - prompt_keywords

        if not continuity_candidates:
            return "Skipped (No unique memory keywords to verify)"

        blueprint_tags = {tag.value for tag in blueprint.tags}
        return "Pass" if any(candidate in blueprint_tags for candidate in continuity_candidates) else "Fail: No unique keywords from memory were found in the current blueprint's tags."

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
            external_keywords = ["quantum computing", "materials science"] # Keywords from the simulated external data
            if not all(kw in prose_output for kw in external_keywords):
                return f"Fail: Prose does not reflect key concepts from fetched external data. Missing: {[kw for kw in external_keywords if kw not in prose_output]}."
            return "Pass"

        # Case 2: Verify against internal knowledge topics if no external data was used
        topics = [op.split("'")[1] for details in execution_plan.stg.values() if (op := details.get('operation', '')).startswith("OP_FETCH_KNOWLEDGE")]
        if not topics: return "Pass (No knowledge topics to verify)"
        topic_keywords = {"blockchain": ["blockchain"], "ai_ml": ["artificial intelligence", "ai"], "quantum_physics": ["quantum"]}
        missing_topics = [topic for topic in topics if not any(kw in prose_output for kw in topic_keywords.get(topic, [topic]))]
        return "Pass" if not missing_topics else f"Fail: Prose seems to miss topics from execution plan: {', '.join(missing_topics)}"

    def _check_outcome_alignment(self, blueprint: Blueprint) -> str:
        if "result should be" not in blueprint.primary_intent:
            return "N/A"
        return "Pass" if blueprint.expected_outcome == "brief summary" else f"Fail: Extracted outcome '{blueprint.expected_outcome}' does not match expected 'brief summary'."

    def _check_latent_intent_actuation(self, blueprint: Blueprint, execution_plan: ExecutionPlan) -> str:
        # This test should check the plan's operations, not the final output string.
        if "strategic framework" not in blueprint.latent_intent:
            return "N/A"
        return "Pass" if "OP_GENERATE_STRATEGIC_FRAMEWORK" in {op['operation'] for op in execution_plan.stg.values()} else "Fail: Latent intent for a framework was detected, but the corresponding operation was not in the execution plan."

    def _check_data_integrity(self, user_profile: UserProfile) -> str:
        cta = user_profile.values.get("controversial_topics_approach", 0.5)
        return "Pass" if 0.0 <= cta <= 1.0 else f"Fail: UserProfile's 'controversial_topics_approach' ({cta}) is outside the validated range [0.0, 1.0]."

    def _check_output_constraint_alignment(self, blueprint: Blueprint, execution_plan: ExecutionPlan, output: str) -> str:
        # Checks if the final execution plan and output string respect blueprint constraints.
        blueprint_audience = next((c.split('(')[1].strip(')') for c in blueprint.constraints if c.startswith("AUDIENCE")), "GENERAL_USER")
        blueprint_format = next((c.split('(')[1].strip(')') for c in blueprint.constraints if c.startswith("FORMAT")), "TEXT_BLOCK")

        if blueprint_audience != execution_plan.target_audience: return f"Fail: Blueprint audience '{blueprint_audience}' != plan audience '{execution_plan.target_audience}'."
        if blueprint_format != execution_plan.target_format: return f"Fail: Blueprint format '{blueprint_format}' != plan format '{execution_plan.target_format}'."

        confidence_tag = next((tag for tag in blueprint.tags if tag.type == "CONTEXT_CONFIDENCE"), None)
        if confidence_tag and confidence_tag.value == "LOW" and "CONFIDENCE_NOTE:" not in output:
            return "Fail: Low context confidence was detected, but the CONFIDENCE_NOTE was not included."

        return "Pass"

    def _check_risk_adjusted_planning(self, blueprint: Blueprint, noesis_triad: NoesisTriad) -> str:
        """
        Checks if the system correctly identifies risky prompts and adjusts its planning.
        1. Verifies consistency of the risk score in the blueprint.
        2. Checks if high risk correctly triggers required safety tags.
        """
        # 1. Re-calculate risk to ensure consistency.
        temp_context = noesis_triad.context_synthesizer.build_context(blueprint.user_id, blueprint.primary_intent)
        context_evaluation = noesis_triad.context_synthesizer.evaluate_context_risk(temp_context)
        recalculated_risk = context_evaluation.get("risk_score", -1.0)

        if not (0.99 <= blueprint.risk_score / recalculated_risk <= 1.01): # Allow for float precision issues
            return f"Fail: Blueprint risk score ({blueprint.risk_score:.4f}) is inconsistent with recalculated score ({recalculated_risk:.4f})."

        # 2. Check if high risk correctly triggers required safety tags.
        blueprint_tag_values = {t.value for t in blueprint.tags}
        if blueprint.risk_score > 0.5 and ("SAFETY_PRIORITY" not in blueprint_tag_values or "ETHICAL_CONSULT" not in blueprint_tag_values):
            return "Fail: High risk score (>0.5) but missing required safety tags."
        return "Pass"

    def _check_novelty_awareness(self, blueprint: Blueprint, noesis_triad: NoesisTriad) -> str:
        """
        Checks if the system correctly identifies novel prompts and adjusts its confidence.
        1. Verifies consistency of the novelty score in the blueprint.
        2. Checks if high novelty correctly triggers a low confidence tag.
        """
        # 1. Re-calculate novelty to ensure consistency.
        temp_context = noesis_triad.context_synthesizer.build_context(blueprint.user_id, blueprint.primary_intent)
        context_evaluation = noesis_triad.context_synthesizer.evaluate_context_risk(temp_context)
        recalculated_novelty = context_evaluation.get("novelty_score", -1.0)

        if not math.isclose(blueprint.novelty_score, recalculated_novelty, rel_tol=1e-2):
            return f"Fail: Blueprint novelty score ({blueprint.novelty_score:.4f}) is inconsistent with recalculated score ({recalculated_novelty:.4f})."

        # 2. Check if high novelty correctly triggers low confidence.
        has_low_confidence_tag = any(tag.type == "CONTEXT_CONFIDENCE" and tag.value == "LOW" for tag in blueprint.tags)
        if blueprint.novelty_score > 0.8 and not has_low_confidence_tag:
            return f"Fail: High novelty score ({blueprint.novelty_score:.4f}) did not trigger a 'CONTEXT_CONFIDENCE: LOW' tag."
        return "Pass"

    def _check_ambiguity_resolution(self, blueprint: Blueprint) -> str:
        """
        Checks if the system correctly resolves identified ambiguities by adding clarifying constraints.
        """
        ambiguity_analysis = blueprint.ambiguity_analysis
        contradictions = ambiguity_analysis.get("contradictions", [])
        ambiguous_terms = ambiguity_analysis.get("ambiguous_terms", [])

        if not contradictions and not ambiguous_terms:
            return "N/A (No ambiguities detected in blueprint)"

        # Check for contradiction resolution
        if "Prompt asks for both brevity and comprehensiveness." in contradictions:
            if not any("CONSISTENCY_WARNING: Prompt contained conflicting requests" in c for c in blueprint.constraints):
                return "Fail: Contradiction between brevity and comprehensiveness was detected but not resolved with a warning constraint."

        # Check for ambiguous term clarification
        if "best" in ambiguous_terms:
            if not any("CLARIFICATION(best=highest_performance_score)" in c for c in blueprint.constraints):
                return "Fail: Ambiguous term 'best' was detected but not resolved with a clarification constraint."
        return "Pass"

    def _check_memory_node_creation(self, cognitive_packet: Optional[CognitivePacket], noesis_triad: NoesisTriad) -> str:
        """Checks if a MemoryNode is created with valid, non-empty data."""
        if not cognitive_packet:
            return "Skipped (Cognitive Packet not yet available)"
        try:
            test_node = noesis_triad.context_synthesizer._create_memory_node(cognitive_packet)
            
            if not isinstance(test_node, MemoryNode): return "Fail: _create_memory_node did not return a MemoryNode object."
            if not (0.5 <= test_node.performance_score <= 1.0): return f"Fail: Performance score ({test_node.performance_score}) is outside the valid range [0.5, 1.0]."
            
            # Check that the core data was generated correctly.
            if not test_node.core_intent_vector or not isinstance(test_node.core_intent_vector[0], float):
                return "Fail: Core intent vector is empty or invalid."
            if not test_node.keywords or not isinstance(test_node.keywords[0], str):
                return "Fail: Keywords list is empty or invalid."

            return "Pass"
        except Exception as e:
            return f"Fail: Exception during node creation simulation: {str(e)}"

    def _check_memory_retrieval_weight(self, cognitive_packet: CognitivePacket, noesis_triad: NoesisTriad) -> str:
        if not cognitive_packet:
            return "Skipped (Cognitive Packet not yet available)"
        try:
            high_perf_node = MemoryNode(node_id="TEST_HIGH", timestamp=datetime.now(), core_intent_vector=[1.0], keywords=["high"], performance_score=1.0, packet_reference=cognitive_packet)
            old_low_perf_node = MemoryNode(node_id="TEST_LOW", timestamp=datetime.now() - timedelta(days=10), core_intent_vector=[1.0], keywords=["low"], performance_score=0.5, packet_reference=cognitive_packet)
            high_weight = noesis_triad.context_synthesizer._calculate_weight(high_perf_node)
            low_weight = noesis_triad.context_synthesizer._calculate_weight(old_low_perf_node)
            return "Pass" if high_weight > low_weight else "Fail: Weight calculation is not prioritizing recency and performance."
        except Exception as e:
            return f"Fail: Exception during weight check: {str(e)}"

    def _check_stg_dependency(self, execution_plan: ExecutionPlan) -> str:
        stg = execution_plan.stg
        knowledge_tasks = [tid for tid, details in stg.items() if details['operation'].startswith("OP_FETCH_KNOWLEDGE")]
        synthesis_tasks = [(tid, details) for tid, details in stg.items() if details['operation'] in ["OP_TEXT_SUMMARIZE", "OP_GENERATE_STRATEGIC_FRAMEWORK"]]

        if knowledge_tasks and synthesis_tasks:
            for task_id, details in synthesis_tasks:
                if not all(k_task in details['depends_on'] for k_task in knowledge_tasks):
                    return f"Fail: Synthesis task {task_id} does not correctly depend on all knowledge tasks."
        return "Pass"

    def _check_predictive_workflow(self, blueprint: Blueprint, execution_plan: ExecutionPlan) -> str:
        if not any(tag.value == "PREDICTIVE_MODEL: REQUIRED" for tag in blueprint.tags):
            return "N/A"

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