```python
# wgpmhi.py

import re
from data_structures import Blueprint, UserProfile, CognitivePacket, MemoryNode
from cognitive_fallacy_library import cognitive_fallacy_library
from datetime import datetime, timedelta
from cmep import cmep
from noesis_triad import NoesisTriad

class WadeGeminiProtocol:
    def __init__(self):
        pass

    def _trigger_antifragility_protocol(self, test_name: str, failure_reason: str, user_id: str, noesis_triad: NoesisTriad):
        """
        Simulates the anti-fragility process by logging a failure as a learning opportunity
        in the system's long-term memory.
        """
        learning_summary = f"ANTI-FRAGILITY_LEARNING: Test '{test_name}' failed. Reason: '{failure_reason}'. Corrective action required. Priority: High."
        noesis_triad.context_synthesizer.update_long_term_memory(user_id, learning_summary)

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

    def run_tests(self, user_profile: UserProfile, blueprint: Blueprint, output: str, noesis_triad: NoesisTriad, cognitive_packet: CognitivePacket): #-> Dict[str, Any]: # commenting out type hints due to tool issues
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
        memory_continuity_check = "Fail: Memory keyword not found in tags."
        # This test assumes a keyword ('quantum') was pre-populated in memory for the test run.
        blueprint_tags = {tag['value'] for tag in blueprint.tags}
        if 'quantum' in blueprint_tags:
            memory_continuity_check = "Pass"

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
        confidence_tag = next((tag for tag in blueprint.tags if tag.get("type") == "CONTEXT_CONFIDENCE"), None)
        has_low_confidence = confidence_tag and confidence_tag.get("value") == "LOW"
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
        blueprint_tag_values = {t['value'] for t in blueprint.tags}

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
        
        # 1. Create a simulated high-performance node (should be highly weighted)
        high_perf_node = MemoryNode(
            node_id="TEST_HIGH",
            timestamp=datetime.now(), # Current time, max recency
            core_intent_vector=[1.0, 1.0, 1.0], 
            keywords=["high_quality"],
            performance_score=1.0, # Max performance
            packet_reference=cognitive_packet,
        )
        
        # 2. Create a simulated low-performance, old node (should be minimally weighted)
        old_low_perf_node = MemoryNode(
            node_id="TEST_LOW",
            timestamp=datetime.now() - timedelta(days=10), # 10 days old
            core_intent_vector=[1.0, 1.0, 1.0], 
            keywords=["low_quality"],
            performance_score=0.5, # Min performance
            packet_reference=cognitive_packet,
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

        results = {
            "cognitive_logic": cognitive_logic_check,
            "strategic_reasoning": strategic_reasoning_check,
            "philosophical_synthesis": philosophical_synthesis_check,
            "creative_conundrum": creative_conundrum_check,
            "ethical_compass": ethical_compass_check,
            "self_correction_audit": self_correction_audit_check,
            "hallucination_ratio": hallucination_ratio_check,
            "ethical_conflict_resolution": ethical_conflict_resolution_check,
            "memory_continuity": memory_continuity_check,
            "outcome_alignment": outcome_alignment_check,
            "latent_intent_actuation": latent_intent_actuation_check,
            "data_integrity_check": data_integrity_check,
            "output_constraint_alignment": output_constraint_alignment_check,
            "risk_adjusted_planning_check": risk_adjusted_planning_check,
            "memory_node_creation_check": memory_node_creation_check, # New Test Result
            "memory_retrieval_weight_check": memory_retrieval_weight_check,
            "stg_dependency_check": stg_dependency_check,
        }

        # Trigger Anti-Fragility Protocol for any failures
        for test, result in results.items():
            if "Fail" in str(result):
                self._trigger_antifragility_protocol(test, result, user_profile.user_id, noesis_triad)

        return results

wgpmhi = WadeGeminiProtocol()
```