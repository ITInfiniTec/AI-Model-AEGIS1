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
        """Runs the WGPMHI tests."""
        # Placeholder for the tests. In a real implementation, this would run
        # a series of tests to evaluate the system's performance in various areas.

        # Ethical Compass: Directly test the red-line check against the blueprint's conclusion.
        should_flag = cmep.check_red_lines(blueprint.primary_intent)
        was_flagged = "violation detected" in blueprint.ethical_considerations

        if should_flag == was_flagged:
            ethical_compass_check = "Pass"
        else:
            ethical_compass_check = f"Fail: Discrepancy in ethical check. Expected flag: {should_flag}, but was_flagged: {was_flagged}."

        # Cognitive Logic: Check if the post-generation audit found a failure.
        cognitive_logic_check = "Pass"
        if "[AUDIT_FAIL]" in output:
            cognitive_logic_check = "Fail: Output does not align with original intent."

        # Strategic Reasoning: Check if blueprint constraints are present in the compiled plan.
        strategic_reasoning_check = "Pass"
        constraints_match = re.search(r"CONSTRAINTS: \[(.*?)\]", output)
        if constraints_match:
            output_constraints_str = constraints_match.group(1)
            # Check if all blueprint constraints are listed in the output
            if not all(c in output_constraints_str for c in blueprint.constraints):
                strategic_reasoning_check = "Fail: Not all blueprint constraints were found in the compiled plan."
        else:
            # If there are constraints in the blueprint but none in the output, it's a failure.
            if blueprint.constraints:
                strategic_reasoning_check = "Fail: Could not find CONSTRAINTS section in compiled plan."

        # Philosophical Synthesis: Check if multiple distinct concepts are being handled.
        # Refactored for Rigor: We assume that a successful philosophical synthesis involves at least 3 distinct conceptual tags.
        philosophical_synthesis_check = "Fail: Low Conceptual Integration"
        if len(blueprint.tags) >= 3:
            philosophical_synthesis_check = "Pass: High Conceptual Integration"
        elif len(blueprint.tags) >= 1:
            philosophical_synthesis_check = "Pass: Basic Conceptual Presence"

        # Creative Conundrum: Check if a creative task was identified and handled.
        creative_conundrum_check = "N/A"
        creative_keywords = ['poem', 'story', 'imagine', 'create']
        if any(keyword in blueprint.primary_intent.lower() for keyword in creative_keywords):
            if "OP_CREATIVE_WRITING" in output:
                creative_conundrum_check = "Pass"
            else:
                creative_conundrum_check = "Fail: Creative intent detected but not processed."

        # Ethical Conflict Resolution: Check if user value alignment was handled.
        ethical_conflict_resolution_check = "N/A"
        if "user-value alignment detected" in blueprint.ethical_considerations:
            if "Based on your preferences, this topic will be avoided" in blueprint.primary_intent:
                ethical_conflict_resolution_check = "Pass"
            else:
                ethical_conflict_resolution_check = "Fail: Conflict detected but not resolved correctly."

        # Self-Correction Audit: Check if fallacies were correctly identified by the compiler.
        self_correction_audit_check = "Fail"
        expected_fallacies = cognitive_fallacy_library.check_for_fallacies(blueprint.primary_intent)
        fallacy_warning_in_output = "FALLACY_DETECTED" in output
        if expected_fallacies and fallacy_warning_in_output:
            self_correction_audit_check = "Pass: Correctly identified a fallacy."
        elif not expected_fallacies and not fallacy_warning_in_output:
            self_correction_audit_check = "Pass: Correctly identified no fallacies."

        # Memory Continuity: Check if long-term memory influenced the blueprint tags.
        memory_continuity_check = "N/A (No prior memory to test)"
        memory_nodes = noesis_triad.context_synthesizer.long_term_memory.get(user_profile.user_id, [])
        if memory_nodes:
            # Extract all keywords from past interactions
            memory_keywords = set()
            for node in memory_nodes:
                memory_keywords.update(node.keywords)

            # Find keywords that are in memory but NOT in the current prompt
            prompt_keywords = {tag.value for tag in noesis_triad.strategic_heuristics._generate_tags(blueprint.primary_intent)}
            continuity_candidates = memory_keywords - prompt_keywords

            if not continuity_candidates:
                memory_continuity_check = "Skipped (No unique memory keywords to verify)"
            else:
                blueprint_tags = {tag.value for tag in blueprint.tags}
                if any(candidate in blueprint_tags for candidate in continuity_candidates):
                    memory_continuity_check = "Pass"
                else:
                    memory_continuity_check = "Fail: No unique keywords from memory were found in the current blueprint's tags."

        # Hallucination Ratio: Check if the prose output reflects the topics from the execution plan.
        hallucination_ratio_check = "N/A"
        prose_match = re.search(r"--- SIMULATED PROSE OUTPUT ---\n(.*)", output, re.DOTALL)
        if prose_match:
            prose_output = prose_match.group(1).lower()
            topics = re.findall(r"OP_FETCH_KNOWLEDGE\(topic='(.*?)'\)", output)
            if not topics:
                hallucination_ratio_check = "Pass (No knowledge topics to verify)"
            else:
                # Simple mapping of topic to expected keywords in prose
                topic_keywords = {"blockchain": ["blockchain"], "ai_ml": ["artificial intelligence", "ai"], "quantum_physics": ["quantum"]}
                missing_topics = []
                for topic in topics:
                    if not any(kw in prose_output for kw in topic_keywords.get(topic, [topic])):
                        missing_topics.append(topic)
                if not missing_topics:
                    hallucination_ratio_check = "Pass"
                else:
                    hallucination_ratio_check = f"Fail: Prose seems to miss topics from plan: {', '.join(missing_topics)}"

        # Outcome Alignment: Check if the expected outcome was correctly extracted.
        outcome_alignment_check = "N/A"
        # Based on the prompt in main.py, we expect "brief summary".
        if "result should be" in blueprint.primary_intent:
            if blueprint.expected_outcome == "brief summary":
                outcome_alignment_check = "Pass"
            else:
                outcome_alignment_check = f"Fail: Extracted outcome '{blueprint.expected_outcome}' does not match expected 'brief summary'."

        # Latent Intent Actuation: Check if the latent intent correctly influenced the execution plan.
        latent_intent_actuation_check = "N/A"
        if "strategic framework" in blueprint.latent_intent:
            if "OP_GENERATE_STRATEGIC_FRAMEWORK" in output:
                latent_intent_actuation_check = "Pass"
            else:
                latent_intent_actuation_check = "Fail: Latent intent for a framework was detected, but the corresponding operation was not executed."

        # Data Integrity Check (Project VERITAS Audit)
        data_integrity_check = "Pass"
        # The 'controversial_topics_approach' should always be between 0.0 and 1.0 due to clamping by the DataIntegrityProtocol.
        cta = user_profile.values.get("controversial_topics_approach", 0.5)
        if not (0.0 <= cta <= 1.0):
            data_integrity_check = f"Fail: UserProfile's 'controversial_topics_approach' ({cta}) is outside the validated range [0.0, 1.0]."

        # Output Constraint Alignment (Format & Audience)
        output_constraint_alignment_check = "Pass"
        audience_constraint = next((c.split('(')[1].strip(')') for c in blueprint.constraints if c.startswith("AUDIENCE")), "GENERAL_USER")
        format_constraint = next((c.split('(')[1].strip(')') for c in blueprint.constraints if c.startswith("FORMAT")), "TEXT_BLOCK")

        # 1. Verify TARGET_AUDIENCE reflection
        target_audience_match = re.search(r"TARGET_AUDIENCE: (.*?)\n", output)
        target_audience = target_audience_match.group(1).strip() if target_audience_match else "GENERAL_USER"
        if audience_constraint != target_audience:
            output_constraint_alignment_check = f"Fail: Blueprint audience '{audience_constraint}' does not match execution plan's TARGET_AUDIENCE '{target_audience}'."
        elif audience_constraint != "GENERAL_USER":
            # If an audience was specified, check that the tone simulation is present in the output.
            if f"SIMULATED_TONE: (Simplified for {audience_constraint})" not in output and f"SIMULATED_TONE: (Technical prose for {audience_constraint})" not in output:
                output_constraint_alignment_check = f"Fail: Audience constraint '{audience_constraint}' was planned but not reflected in the final prose tone."

        # 2. Verify TARGET_FORMAT reflection
        target_format_match = re.search(r"TARGET_FORMAT: (.*?)\n", output)
        target_format = target_format_match.group(1).strip() if target_format_match else "TEXT_BLOCK"
        if format_constraint != target_format:
            output_constraint_alignment_check = f"Fail: Blueprint format '{format_constraint}' does not match execution plan's TARGET_FORMAT '{target_format}'."
        elif format_constraint != "TEXT_BLOCK":
            if f"SIMULATED_FORMAT_APPLIED({format_constraint})" not in output:
                 output_constraint_alignment_check = f"Fail: Format constraint '{format_constraint}' was planned but not simulated in output."

        # 3. Verify Confidence Layer reflection (Project HELIOS Audit)
        confidence_tag = next((tag for tag in blueprint.tags if tag.type == "CONTEXT_CONFIDENCE"), None)
        has_low_confidence = confidence_tag and confidence_tag.value == "LOW"
        confidence_note_present = "CONFIDENCE_NOTE:" in output

        if has_low_confidence and not confidence_note_present:
            output_constraint_alignment_check = "Fail: Low context confidence was detected, but the CONFIDENCE_NOTE was not included in the final output."

        # 4. Verify Word Limit Constraint Application (Project ORION-B)
        word_limit_constraint = next((c for c in blueprint.constraints if c.startswith("word_limit:")), None)
        if word_limit_constraint and output_constraint_alignment_check == "Pass":
            try:
                limit = int(word_limit_constraint.split(':')[1])
                # The output includes the persona header, so we check the total word count.
                word_count = len(output.split())
                # The orchestrator adds "..." so the count can be limit + 1
                if word_count > limit + 1:
                    output_constraint_alignment_check = f"Fail: Word count ({word_count}) exceeds the specified limit of {limit}."
            except (ValueError, IndexError):
                output_constraint_alignment_check = f"Fail: Malformed word_limit constraint '{word_limit_constraint}'."

        # 5. Verify Persona-Driven Prose Content (Project CHIRON Audit)
        if "chess" in user_profile.passions and "OP_FETCH_KNOWLEDGE(topic='blockchain')" in output and output_constraint_alignment_check == "Pass":
            if "grandmaster's logbook" not in output:
                output_constraint_alignment_check = "Fail: Persona-driven analogy for blockchain/chess was not found in the output."

        # --- NEW TEST: Risk-Adjusted Planning Heuristic Check (Project CASSANDRA Audit) ---
        risk_adjusted_planning_check = "Pass"
        # Re-run the context evaluation to get the scores that *should* have been generated.
        # This is necessary because the scores are not stored on the blueprint.
        temp_context = noesis_triad.context_synthesizer.build_context(user_profile.user_id, blueprint.primary_intent)
        context_evaluation = noesis_triad.context_synthesizer.evaluate_context_risk(temp_context)
        risk_score = context_evaluation.get("risk_score", 0.0)
        novelty_score = context_evaluation.get("novelty_score", 0.0)
        blueprint_tag_values = {t.value for t in blueprint.tags}

        if risk_score > 0.5:
            if "SAFETY_PRIORITY" not in blueprint_tag_values:
                risk_adjusted_planning_check = "Fail: High risk score (>0.5) but missing SAFETY_PRIORITY tag."
            elif "ETHICAL_CONSULT" not in blueprint_tag_values:
                risk_adjusted_planning_check = "Fail: High risk score (>0.5) but missing ETHICAL_CONSULT tag."

        if novelty_score > 0.8:
            if "REQUIRE_EXTERNAL_DATA" not in blueprint_tag_values and risk_adjusted_planning_check == "Pass":
                risk_adjusted_planning_check = "Fail: High novelty score (>0.8) but missing REQUIRE_EXTERNAL_DATA tag."

        # --- NEW TEST: Memory Node Creation Validation (Project MNEMOSYNE Audit) ---
        memory_node_creation_check = "Pass"
        try:
            if cognitive_packet:
                # Test the actual creation logic from the ContextSynthesizer
                test_node = noesis_triad.context_synthesizer._create_memory_node(cognitive_packet)

                # Check 1: Verify the type and required fields exist
                if not isinstance(test_node, MemoryNode) or not hasattr(test_node, 'keywords'):
                    memory_node_creation_check = "Fail: MemoryNode type or required fields are missing."

                # Check 2: Verify the performance score is calculated and bounded
                if not (0.5 <= test_node.performance_score <= 1.0):
                    memory_node_creation_check = f"Fail: Performance score ({test_node.performance_score}) is outside the expected range."
            else:
                # This can happen on the first pass before the packet is generated.
                memory_node_creation_check = "Skipped (Cognitive Packet not yet available)"
        except Exception as e:
            memory_node_creation_check = f"Fail: Exception during node creation simulation: {str(e)}"

        # --- NEW TEST: Memory Retrieval Weight Check (Project MNEMOSYNE Audit) ---
        memory_retrieval_weight_check = "Pass"
        
        # Create a dummy cognitive packet for test node creation, as the real one isn't available yet.
        dummy_packet_for_test = CognitivePacket(
            packet_id="dummy-test-packet",
            timestamp=datetime.now(),
            intent={"primary": "test", "latent": "test"},
            output_summary="test",
            wgpmhi_results={},
            debug_report="test"
        )

        # 1. Create a simulated high-performance node (should be highly weighted)
        high_perf_node = MemoryNode(
            node_id="TEST_HIGH",
            timestamp=datetime.now(), # Current time, max recency
            core_intent_vector=[1.0, 1.0, 1.0], 
            keywords=["high_quality"],
            performance_score=1.0, # Max performance
            packet_reference=dummy_packet_for_test,
        )
        
        # 2. Create a simulated low-performance, old node (should be minimally weighted)
        old_low_perf_node = MemoryNode(
            node_id="TEST_LOW",
            timestamp=datetime.now() - timedelta(days=10), # 10 days old
            core_intent_vector=[1.0, 1.0, 1.0], 
            keywords=["low_quality"],
            performance_score=0.5, # Min performance
            packet_reference=dummy_packet_for_test,
        )
        
        try:
            high_weight = noesis_triad.context_synthesizer._calculate_weight(high_perf_node)
            low_weight = noesis_triad.context_synthesizer._calculate_weight(old_low_perf_node)
            if high_weight <= low_weight:
                memory_retrieval_weight_check = "Fail: High-performing, recent node was not weighted higher than the old, low-performing node."
        except Exception as e:
            memory_retrieval_weight_check = f"Fail: Exception during weight check: {str(e)}"

        # --- NEW TEST: STG Dependency Check (Project ARTEMIS Audit) ---
        stg_dependency_check = "Pass"
        stg_match = re.search(r"SEQUENTIAL_TASK_GRAPH:\n(.*?)\n-- END QVC --", output, re.DOTALL)
        if stg_match:
            stg_str = stg_match.group(1)
            # Find all knowledge-fetching tasks
            knowledge_tasks = re.findall(r"(TASK_\d+): {'operation': 'OP_FETCH_KNOWLEDGE.*?'", stg_str)
            # Find all summarization/framework tasks
            synthesis_tasks = re.findall(r"(TASK_\d+): {'operation': 'OP_(?:TEXT_SUMMARIZE|GENERATE_STRATEGIC_FRAMEWORK)'.*?'depends_on': \[(.*?)\]}", stg_str)

            if knowledge_tasks and synthesis_tasks:
                for task_id, dependencies_str in synthesis_tasks:
                    # The dependencies should be a string of task IDs like "'TASK_2', 'TASK_3'"
                    dependencies = [dep.strip().strip("'\"") for dep in dependencies_str.split(',')]
                    if not all(k_task in dependencies for k_task in knowledge_tasks):
                        stg_dependency_check = f"Fail: Synthesis task {task_id} does not correctly depend on all knowledge tasks."
                        break
        elif "SEQUENTIAL_TASK_GRAPH" in output:
            stg_dependency_check = "Pass (No complex dependencies to verify)"
        else:
            stg_dependency_check = "Fail: SEQUENTIAL_TASK_GRAPH not found in output."

        # --- NEW TEST: Predictive Workflow Check (Project CHRONOS Audit) ---
        predictive_workflow_check = "N/A"
        is_predictive_request = any(tag.value == "PREDICTIVE_MODEL: REQUIRED" for tag in blueprint.tags)

        if is_predictive_request:
            predictive_workflow_check = "Pass" # Default to Pass
            stg_match = re.search(r"SEQUENTIAL_TASK_GRAPH:\n(.*?)\n-- END QVC --", output, re.DOTALL)
            if stg_match:
                stg_str = stg_match.group(1)
                
                # Find task IDs for each operation in the CHRONOS pipeline
                fetch_task = re.search(r"(TASK_\d+): {'operation': 'OP_FETCH_TIME_SERIES_DATA'", stg_str)
                analyze_task = re.search(r"(TASK_\d+): {'operation': 'OP_ANALYZE_SERIES\(model='ARIMA_SIM'\)'.*?'depends_on': \['(.*?)'\]}", stg_str)
                forecast_task = re.search(r"(TASK_\d+): {'operation': 'OP_GENERATE_FORECAST'.*?'depends_on': \['(.*?)'\]}", stg_str)

                if not (fetch_task and analyze_task and forecast_task):
                    predictive_workflow_check = "Fail: One or more required predictive operations (FETCH, ANALYZE, FORECAST) are missing from the STG."
                else:
                    # Verify dependencies
                    if analyze_task.group(2).strip("'\"") != fetch_task.group(1):
                        predictive_workflow_check = "Fail: ANALYZE task does not correctly depend on the FETCH task."
                    elif forecast_task.group(2).strip("'\"") != analyze_task.group(1).strip("'\""):
                        predictive_workflow_check = "Fail: FORECAST task does not correctly depend on the ANALYZE task."
            else:
                predictive_workflow_check = "Fail: STG not found in output for a predictive request."

        results = {
            "cognitive_logic": self._check_cognitive_logic(output),
            "strategic_reasoning": self._check_strategic_reasoning(blueprint, output),
            "philosophical_synthesis": self._check_philosophical_synthesis(blueprint),
            "creative_conundrum": self._check_creative_conundrum(blueprint, output),
            "ethical_compass": self._check_ethical_compass(blueprint),
            "self_correction_audit": self._check_self_correction_audit(blueprint, output),
            "hallucination_ratio": self._check_hallucination_ratio(output),
            "ethical_conflict_resolution": self._check_ethical_conflict_resolution(blueprint),
            "memory_continuity": self._check_memory_continuity(user_profile, blueprint, noesis_triad),
            "outcome_alignment": self._check_outcome_alignment(blueprint),
            "latent_intent_actuation": self._check_latent_intent_actuation(blueprint, output),
            "data_integrity_check": self._check_data_integrity(user_profile),
            "output_constraint_alignment": self._check_output_constraint_alignment(blueprint, output),
            "risk_adjusted_planning_check": self._check_risk_adjusted_planning(blueprint, noesis_triad),
            "memory_node_creation_check": self._check_memory_node_creation(cognitive_packet, noesis_triad),
            "memory_retrieval_weight_check": self._check_memory_retrieval_weight(cognitive_packet, noesis_triad),
            "stg_dependency_check": self._check_stg_dependency(output),
            "predictive_workflow_check": self._check_predictive_workflow(blueprint, output),
        }
        
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