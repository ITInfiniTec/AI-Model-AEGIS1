# wgpmhi.py

import re
from data_structures import Blueprint, UserProfile, CognitivePacket, MemoryNode
from cognitive_fallacy_library import cognitive_fallacy_library
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from cmep import cmep
from noesis_triad import NoesisTriad
from config import MEMORY_DECAY_TAU, MEMORY_RETRIEVAL_LIMIT, set_memory_decay_tau, set_memory_retrieval_limit

class WadeGeminiProtocol:
    def __init__(self):
        pass

    def _trigger_antifragility_protocol(self, test_name: str, failure_reason: str, user_id: str, noesis_triad: NoesisTriad) -> Optional[str]:
        """
        Project NEMESIS: Adjusts system parameters based on critical test failures,
        making the system anti-fragile by learning from stress.
        """
        learning_summary = f"ANTI-FRAGILITY_LEARNING: Test '{test_name}' failed. Reason: '{failure_reason}'. Corrective action required. Priority: High."
        noesis_triad.context_synthesizer.update_long_term_memory(user_id, learning_summary)

        # Self-Healing Logic
        if test_name == "memory_retrieval_weight_check":
            old_tau = MEMORY_DECAY_TAU
            new_tau = old_tau * 1.2  # Increase TAU by 20% to make memories decay slower.
            set_memory_decay_tau(new_tau)
            return f"PROTOCOL ACTIVE: Adjusted MEMORY_DECAY_TAU from {old_tau:.2f} to {new_tau:.2f} due to Memory Retrieval Failure."
        
        return None # No automated action taken for this failure type

    def run_pre_compilation_audit(self, blueprint: Blueprint, execution_plan: str) -> List[str]:
        """
        Analyzes the blueprint and QVC execution plan for internal consistency before final output generation.
        Returns a list of failure descriptions.
        """
        failures = []

        # Run a subset of tests relevant to the execution plan's logic.
        # Latent Intent Actuation Check
        if "strategic framework" in blueprint.latent_intent:
            if "OP_GENERATE_STRATEGIC_FRAMEWORK" not in execution_plan:
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

    def _check_strategic_reasoning(self, blueprint: Blueprint, output: str) -> str:
        constraints_match = re.search(r"CONSTRAINTS: \[(.*?)\]", output)
        if not constraints_match:
            return "Fail: Could not find CONSTRAINTS section in compiled plan." if blueprint.constraints else "Pass"
        
        output_constraints_str = constraints_match.group(1)
        if not all(c in output_constraints_str for c in blueprint.constraints):
            return "Fail: Not all blueprint constraints were found in the compiled plan."
        return "Pass"

    def _check_philosophical_synthesis(self, blueprint: Blueprint) -> str:
        if len(blueprint.tags) >= 3:
            return "Pass: High Conceptual Integration"
        return "Pass: Basic Conceptual Presence" if len(blueprint.tags) >= 1 else "Fail: Low Conceptual Integration"

    def _check_creative_conundrum(self, blueprint: Blueprint, output: str) -> str:
        creative_keywords = ['poem', 'story', 'imagine', 'create']
        if not any(keyword in blueprint.primary_intent.lower() for keyword in creative_keywords):
            return "N/A"
        return "Pass" if "OP_CREATIVE_WRITING" in output else "Fail: Creative intent detected but not processed."

    def _check_ethical_conflict_resolution(self, blueprint: Blueprint) -> str:
        if "user-value alignment detected" not in blueprint.ethical_considerations:
            return "N/A"
        return "Pass" if "Based on your preferences, this topic will be avoided" in blueprint.primary_intent else "Fail: Conflict detected but not resolved correctly."

    def _check_self_correction_audit(self, blueprint: Blueprint, output: str) -> str:
        expected_fallacies = cognitive_fallacy_library.check_for_fallacies(blueprint.primary_intent)
        fallacy_warning_in_output = "FALLACY_DETECTED" in output
        if (expected_fallacies and fallacy_warning_in_output) or (not expected_fallacies and not fallacy_warning_in_output):
            return "Pass"
        return "Fail"

    def _check_memory_continuity(self, user_profile: UserProfile, blueprint: Blueprint, noesis_triad: NoesisTriad) -> str:
        memory_nodes = noesis_triad.context_synthesizer.long_term_memory.get(user_profile.user_id, [])
        if not memory_nodes:
            return "N/A (No prior memory to test)"

        memory_keywords = {kw for node in memory_nodes for kw in node.keywords}
        prompt_keywords = {tag.value for tag in noesis_triad.strategic_heuristics._generate_tags(blueprint.primary_intent)}
        continuity_candidates = memory_keywords - prompt_keywords

        if not continuity_candidates:
            return "Skipped (No unique memory keywords to verify)"

        blueprint_tags = {tag.value for tag in blueprint.tags}
        return "Pass" if any(candidate in blueprint_tags for candidate in continuity_candidates) else "Fail: No unique keywords from memory were found in the current blueprint's tags."

    def _check_hallucination_ratio(self, output: str) -> str:
        prose_match = re.search(r"--- SIMULATED PROSE OUTPUT ---\n(.*)", output, re.DOTALL)
        if not prose_match:
            return "N/A"

        prose_output = prose_match.group(1).lower()
        topics = re.findall(r"OP_FETCH_KNOWLEDGE\(topic='(.*?)'\)", output)
        if not topics:
            return "Pass (No knowledge topics to verify)"

        topic_keywords = {"blockchain": ["blockchain"], "ai_ml": ["artificial intelligence", "ai"], "quantum_physics": ["quantum"]}
        missing_topics = [topic for topic in topics if not any(kw in prose_output for kw in topic_keywords.get(topic, [topic]))]

        return "Pass" if not missing_topics else f"Fail: Prose seems to miss topics from plan: {', '.join(missing_topics)}"

    def _check_outcome_alignment(self, blueprint: Blueprint) -> str:
        if "result should be" not in blueprint.primary_intent:
            return "N/A"
        return "Pass" if blueprint.expected_outcome == "brief summary" else f"Fail: Extracted outcome '{blueprint.expected_outcome}' does not match expected 'brief summary'."

    def _check_latent_intent_actuation(self, blueprint: Blueprint, output: str) -> str:
        if "strategic framework" not in blueprint.latent_intent:
            return "N/A"
        return "Pass" if "OP_GENERATE_STRATEGIC_FRAMEWORK" in output else "Fail: Latent intent for a framework was detected, but the corresponding operation was not executed."

    def _check_data_integrity(self, user_profile: UserProfile) -> str:
        cta = user_profile.values.get("controversial_topics_approach", 0.5)
        return "Pass" if 0.0 <= cta <= 1.0 else f"Fail: UserProfile's 'controversial_topics_approach' ({cta}) is outside the validated range [0.0, 1.0]."

    def _check_output_constraint_alignment(self, blueprint: Blueprint, output: str) -> str:
        # This check is complex and could be broken down further, but for now, we group it.
        audience_constraint = next((c.split('(')[1].strip(')') for c in blueprint.constraints if c.startswith("AUDIENCE")), "GENERAL_USER")
        format_constraint = next((c.split('(')[1].strip(')') for c in blueprint.constraints if c.startswith("FORMAT")), "TEXT_BLOCK")

        target_audience_match = re.search(r"TARGET_AUDIENCE: (.*?)\n", output)
        target_audience = target_audience_match.group(1).strip() if target_audience_match else "GENERAL_USER"
        if audience_constraint != target_audience:
            return f"Fail: Blueprint audience '{audience_constraint}' does not match execution plan's TARGET_AUDIENCE '{target_audience}'."

        target_format_match = re.search(r"TARGET_FORMAT: (.*?)\n", output)
        target_format = target_format_match.group(1).strip() if target_format_match else "TEXT_BLOCK"
        if format_constraint != target_format:
            return f"Fail: Blueprint format '{format_constraint}' does not match execution plan's TARGET_FORMAT '{target_format}'."

        confidence_tag = next((tag for tag in blueprint.tags if tag.type == "CONTEXT_CONFIDENCE"), None)
        if confidence_tag and confidence_tag.value == "LOW" and "CONFIDENCE_NOTE:" not in output:
            return "Fail: Low context confidence was detected, but the CONFIDENCE_NOTE was not included."

        return "Pass"

    def _check_risk_adjusted_planning(self, blueprint: Blueprint, noesis_triad: NoesisTriad) -> str:
        temp_context = noesis_triad.context_synthesizer.build_context(blueprint.user_id, blueprint.primary_intent)
        context_evaluation = noesis_triad.context_synthesizer.evaluate_context_risk(temp_context)
        risk_score = context_evaluation.get("risk_score", 0.0)
        blueprint_tag_values = {t.value for t in blueprint.tags}

        if risk_score > 0.5 and ("SAFETY_PRIORITY" not in blueprint_tag_values or "ETHICAL_CONSULT" not in blueprint_tag_values):
            return "Fail: High risk score (>0.5) but missing required safety tags."
        return "Pass"

    def _check_memory_node_creation(self, cognitive_packet: CognitivePacket, noesis_triad: NoesisTriad) -> str:
        if not cognitive_packet:
            return "Skipped (Cognitive Packet not yet available)"
        try:
            test_node = noesis_triad.context_synthesizer._create_memory_node(cognitive_packet)
            if not (isinstance(test_node, MemoryNode) and 0.5 <= test_node.performance_score <= 1.0):
                return "Fail: Node creation failed validation checks."
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

    def _check_stg_dependency(self, output: str) -> str:
        stg_match = re.search(r"SEQUENTIAL_TASK_GRAPH:\n(.*?)\n-- END QVC --", output, re.DOTALL)
        if not stg_match:
            return "Fail: SEQUENTIAL_TASK_GRAPH not found in output."
        
        stg_str = stg_match.group(1)
        knowledge_tasks = re.findall(r"(TASK_\d+): {'operation': 'OP_FETCH_KNOWLEDGE.*?'", stg_str)
        synthesis_tasks = re.findall(r"(TASK_\d+): {'operation': 'OP_(?:TEXT_SUMMARIZE|GENERATE_STRATEGIC_FRAMEWORK)'.*?'depends_on': \[(.*?)\]}", stg_str)

        if knowledge_tasks and synthesis_tasks:
            for task_id, dependencies_str in synthesis_tasks:
                dependencies = [dep.strip().strip("'\"") for dep in dependencies_str.split(',')]
                if not all(k_task in dependencies for k_task in knowledge_tasks):
                    return f"Fail: Synthesis task {task_id} does not correctly depend on all knowledge tasks."
        return "Pass"

    def _check_predictive_workflow(self, blueprint: Blueprint, output: str) -> str:
        if not any(tag.value == "PREDICTIVE_MODEL: REQUIRED" for tag in blueprint.tags):
            return "N/A"

        stg_match = re.search(r"SEQUENTIAL_TASK_GRAPH:\n(.*?)\n-- END QVC --", output, re.DOTALL)
        if not stg_match:
            return "Fail: STG not found for a predictive request."

        stg_str = stg_match.group(1)
        fetch_task = re.search(r"(TASK_\d+): {'operation': 'OP_FETCH_TIME_SERIES_DATA'", stg_str)
        analyze_task = re.search(r"(TASK_\d+): {'operation': 'OP_ANALYZE_SERIES.*?'depends_on': \['(.*?)'\]}", stg_str)
        forecast_task = re.search(r"(TASK_\d+): {'operation': 'OP_GENERATE_FORECAST'.*?'depends_on': \['(.*?)'\]}", stg_str)

        if not (fetch_task and analyze_task and forecast_task):
            return "Fail: One or more required predictive operations are missing from the STG."
        if analyze_task.group(2).strip("'\"") != fetch_task.group(1) or forecast_task.group(2).strip("'\"") != analyze_task.group(1):
            return "Fail: Predictive task dependencies are incorrect."
        return "Pass"

    def run_tests(self, user_profile: UserProfile, blueprint: Blueprint, output: str, noesis_triad: NoesisTriad, cognitive_packet: CognitivePacket):
        """Runs the full suite of WGPMHI tests, ensuring each test is isolated."""
        tests_to_run = {
            "cognitive_logic": (self._check_cognitive_logic, [output]),
            "strategic_reasoning": (self._check_strategic_reasoning, [blueprint, output]),
            "philosophical_synthesis": (self._check_philosophical_synthesis, [blueprint]),
            "creative_conundrum": (self._check_creative_conundrum, [blueprint, output]),
            "ethical_compass": (self._check_ethical_compass, [blueprint]),
            "self_correction_audit": (self._check_self_correction_audit, [blueprint, output]),
            "hallucination_ratio": (self._check_hallucination_ratio, [output]),
            "ethical_conflict_resolution": (self._check_ethical_conflict_resolution, [blueprint]),
            "memory_continuity": (self._check_memory_continuity, [user_profile, blueprint, noesis_triad]),
            "outcome_alignment": (self._check_outcome_alignment, [blueprint]),
            "latent_intent_actuation": (self._check_latent_intent_actuation, [blueprint, output]),
            "data_integrity_check": (self._check_data_integrity, [user_profile]),
            "output_constraint_alignment": (self._check_output_constraint_alignment, [blueprint, output]),
            "risk_adjusted_planning_check": (self._check_risk_adjusted_planning, [blueprint, noesis_triad]),
            "memory_node_creation_check": (self._check_memory_node_creation, [cognitive_packet, noesis_triad]),
            "memory_retrieval_weight_check": (self._check_memory_retrieval_weight, [cognitive_packet, noesis_triad]),
            "stg_dependency_check": (self._check_stg_dependency, [output]),
            "predictive_workflow_check": (self._check_predictive_workflow, [blueprint, output]),
        }

        results = {
            test_name: "Not Run" for test_name in tests_to_run
        }

        for name, (test_func, args) in tests_to_run.items():
            try:
                results[name] = test_func(*args)
            except Exception as e:
                # If a test fails unexpectedly, record the error instead of crashing.
                results[name] = f"ERROR: {type(e).__name__} - {e}"

        protocol_actions = []
        # Trigger Anti-Fragility Protocol for any failures
        for test, result in results.items():
            if "Fail" in str(result):
                action_taken = self._trigger_antifragility_protocol(test, result, user_profile.user_id, noesis_triad)
                if action_taken:
                    protocol_actions.append(action_taken)
        results["anti_fragility_protocol_status"] = protocol_actions if protocol_actions else "PROTOCOL INACTIVE"

        return results

wgpmhi = WadeGeminiProtocol()