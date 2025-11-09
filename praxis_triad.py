# praxis_triad.py

import re
from typing import List, Dict, Any
from data_structures import Blueprint, UserProfile, SemanticTag, ExecutionPlan
from cmep import cmep
from cognitive_fallacy_library import cognitive_fallacy_library
from diagnostic_reporter import DiagnosticReporter
from prometheus_iop import prometheus_iop

class UniversalCompiler:
    def __init__(self):
        pass

    def _translate_tags_to_operations(self, tags: List[SemanticTag], latent_intent: str) -> List[str]:
        """Tier 2 Simulation: Translates tags into a sequence of logical operations."""
        operations = set()
        tag_values = {tag.value for tag in tags}

        # Latent intent can add high-level operations.
        if "strategic framework" in latent_intent:
            operations.add("OP_GENERATE_STRATEGIC_FRAMEWORK")

        # Add operations based on keyword tags.
        if 'summarize' in tag_values:
            # Add summarize only if a more specific generation task isn't already present.
            if "OP_GENERATE_STRATEGIC_FRAMEWORK" not in operations and "OP_CREATIVE_WRITING" not in operations:
                operations.add("OP_TEXT_SUMMARIZE")
        if 'quantum' in tag_values or 'physics' in tag_values:
            operations.add("OP_FETCH_KNOWLEDGE(topic='quantum_physics')")
        if 'blockchain' in tag_values:
            operations.add("OP_FETCH_KNOWLEDGE(topic='blockchain')")
        if any(ai_tag in tag_values for ai_tag in ['ai', 'ml', 'ann', 'gnn']):
            operations.add("OP_FETCH_KNOWLEDGE(topic='ai_ml')")
        if any(creative_tag in tag_values for creative_tag in ['poem', 'story', 'imagine', 'create']):
            operations.add("OP_CREATIVE_WRITING")
        
        # --- Project CHRONOS: New Tag Recognition ---
        if any(tag.type == "MODEL_PROTOCOL" and tag.value == "PREDICTIVE_MODEL: REQUIRED" for tag in tags):
            operations.update(["OP_FETCH_TIME_SERIES_DATA", "OP_ANALYZE_SERIES(model='ARIMA_SIM')", "OP_GENERATE_FORECAST"])

        if not operations:
            operations.add("OP_GENERAL_QUERY")
        
        return sorted(list(operations))

    def _translate_operations_to_execution(self, intent: str, operations: List[str], constraints: List[str], fallacies: List[str], blueprint: Blueprint) -> ExecutionPlan:
        """
        Tier 3 Simulation: Compiles operations into a structured ExecutionPlan object.
        This replaces the brittle string-based QVC format.
        """
        # --- Project PROMETHEUS: Simulating I/O Execution for Forecasting ---
        time_series_data = None
        forecast = 'No Time Series Data Analyzed'
        
        if "OP_FETCH_TIME_SERIES_DATA" in operations:
            # Simulate fetching a specific series based on the intent
            series_id = "NETWORK_TRAFFIC_TRENDS"
            time_series_data = prometheus_iop.fetch_time_series_data(series_id)
            
            if "OP_GENERATE_FORECAST" in operations and time_series_data and time_series_data.data_points:
                last_value = time_series_data.data_points[-1][1]
                forecast = f"High confidence forecast: Next value is projected to be {last_value * 1.05:.2f} (5% growth)."

        # --- Project ARTEMIS: Task Decomposition ---
        task_engine = TaskDecompositionEngine()
        stg = task_engine.generate_stg(operations)

        # --- Constraint Processing for QVC Execution ---
        format_constraint = next((c.split('(')[1].strip(')') for c in constraints if c.startswith("FORMAT")), "TEXT_BLOCK")
        audience_constraint = next((c.split('(')[1].strip(')') for c in constraints if c.startswith("AUDIENCE")), "GENERAL_USER")

        # --- CASSANDRA Tag Processing for QVC Execution ---
        tag_values = {t.value for t in blueprint.tags}
        
        return ExecutionPlan(
            intent_hash=hash(intent),
            constraints=constraints,
            target_format=format_constraint,
            target_audience=audience_constraint,
            fallacy_warnings=fallacies,
            external_data_required="REQUIRE_EXTERNAL_DATA" in tag_values,
            safety_priority="HIGH" if "SAFETY_PRIORITY" in tag_values else "STANDARD",
            ethical_consult_required="ETHICAL_CONSULT" in tag_values,
            simulated_forecast_result=forecast,
            stg=stg,
        )

    def compile_blueprint(self, blueprint: Blueprint) -> ExecutionPlan | str:
        """Compiles the blueprint into an executable output."""
        if "violation detected" in blueprint.ethical_considerations:
            return "Ethical red-line violation detected. Output cannot be generated."
        detected_fallacies = cognitive_fallacy_library.check_for_fallacies(blueprint.primary_intent)
        operations = self._translate_tags_to_operations(blueprint.tags, blueprint.latent_intent)
        return self._translate_operations_to_execution(blueprint.primary_intent, operations, blueprint.constraints, detected_fallacies, blueprint)

class TaskDecompositionEngine:
    """Analyzes operations to generate a Sequential Task Graph (STG)."""
    def generate_stg(self, operations: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Generates a Sequential Task Graph from a list of operations.
        This is a simulation of dependency analysis.
        """
        stg = {}
        task_counter = 1
        knowledge_tasks = []

        for op in operations:
            task_id = f"TASK_{task_counter}"
            if op.startswith("OP_FETCH_KNOWLEDGE"):
                # Critical tasks now have a conditional failure path
                stg[task_id] = {"operation": op, "depends_on": [], "on_failure": f"LOG_ERROR_AND_HALT({task_id})"}
                knowledge_tasks.append(task_id)
            elif op in ["OP_TEXT_SUMMARIZE", "OP_GENERATE_STRATEGIC_FRAMEWORK"]:
                # These operations depend on having knowledge first.
                stg[task_id] = {"operation": op, "depends_on": knowledge_tasks}
            elif op.startswith("OP_FETCH_TIME_SERIES_DATA"):
                stg[task_id] = {"operation": op, "depends_on": [], "on_failure": f"LOG_ERROR_AND_HALT({task_id})"}
                knowledge_tasks.append(task_id) # Treat data fetch as a critical knowledge task
            elif op.startswith("OP_ANALYZE_SERIES"):
                fetch_task = next((tid for tid, details in stg.items() if details["operation"] == "OP_FETCH_TIME_SERIES_DATA"), None)
                stg[task_id] = {"operation": op, "depends_on": [fetch_task] if fetch_task else []}
            elif op == "OP_GENERATE_FORECAST": # type: ignore
                # Forecasting depends on analysis.
                analysis_task = next((tid for tid, details in stg.items() if details["operation"].startswith("OP_ANALYZE_SERIES")), None)
                stg[task_id] = {"operation": op, "depends_on": [analysis_task] if analysis_task else []}
            else:
                stg[task_id] = {"operation": op, "depends_on": []}
            task_counter += 1
        return stg

class ResponseOrchestrator:
    """Applies final formatting, constraints, and confidence layers to the output."""
    def __init__(self):
        # Project CHIRON: A dictionary of analogies to connect topics to user passions.
        self.passion_analogies = {
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
        }

    def _generate_persona_driven_prose(self, user_profile: UserProfile, operations: List[str]) -> str:
        """Generates simulated prose using analogies based on user passions."""
        passions = user_profile.passions
        prose_segments = ["Based on the execution plan, here is a summary of the requested topics."]

        # Map operations to topics more robustly
        topics_in_plan = {op.split("'")[1] for op in operations if op.startswith("OP_FETCH_KNOWLEDGE")}

        for topic in sorted(list(topics_in_plan)): # Sort for consistent output
            analogy_found = False
            for passion in passions:
                if topic in self.passion_analogies and passion in self.passion_analogies[topic]:
                    prose_segments.append(self.passion_analogies[topic][passion])
                    analogy_found = True
                    break  # Use the first matching passion-analogy
            if not analogy_found:
                # Provide a generic fallback if no specific analogy is found for the topic
                prose_segments.append(f"The topic of {topic.replace('_', ' ')} is a complex field with many nuances.")

        return " ".join(prose_segments)

    def orchestrate_response(self, execution_plan: ExecutionPlan, blueprint: Blueprint, user_profile: UserProfile) -> str:
        """Constructs the final output string from the raw execution results and blueprint."""
        constraints = blueprint.constraints
        expected_outcome = blueprint.expected_outcome
        
        # Confidence Layer Integration (Project VESTA)
        confidence_tag = next((tag for tag in blueprint.tags if tag.type == "CONTEXT_CONFIDENCE"), None)
        confidence_statement = ""
        if confidence_tag and confidence_tag.value == "LOW":
            confidence_statement = "CONFIDENCE_NOTE: Context confidence is low. External verification is recommended.\n"

        # Audience Tone Modulation
        tone_header = ""
        if execution_plan.target_audience in ["beginner", "novice"]:
            tone_header = f"SIMULATED_TONE: (Simplified for {execution_plan.target_audience})\n"
        elif execution_plan.target_audience in ["expert", "professional"]:
            tone_header = f"SIMULATED_TONE: (Technical prose for {execution_plan.target_audience})\n"

        # Format Application
        format_tag = ""
        if execution_plan.target_format != "TEXT_BLOCK":
             format_tag = f"SIMULATED_FORMAT_APPLIED({execution_plan.target_format})\n"

        # Expected Outcome Formatting
        output_header = ""
        if expected_outcome == "brief summary":
            output_header = "Here is a brief summary as requested:\n"

        # --- Reconstruct a human-readable version of the plan for the final output ---
        stg_str = "\n".join([f"  {task_id}: {details}" for task_id, details in execution_plan.stg.items()])
        fallacy_warnings = f"\nWARNING: FALLACY_DETECTED({', '.join(execution_plan.fallacy_warnings)}). Proceeding with caution." if execution_plan.fallacy_warnings else ""
        
        plan_str = f"""
-- BEGIN QVC UNFOLDING EXECUTION --
INTENT_HASH: {execution_plan.intent_hash}
CONSTRAINTS: [{', '.join(execution_plan.constraints) if execution_plan.constraints else 'None'}]
TARGET_FORMAT: {execution_plan.target_format}
TARGET_AUDIENCE: {execution_plan.target_audience}{fallacy_warnings}
EXTERNAL_DATA_REQUIRED: {'YES' if execution_plan.external_data_required else 'NO'}
SAFETY_PRIORITY_LEVEL: {execution_plan.safety_priority}
ETHICAL_CONSULTATION: {'YES' if execution_plan.ethical_consult_required else 'NO'}
SIMULATED_FORECAST_RESULT: {execution_plan.simulated_forecast_result}
SEQUENTIAL_TASK_GRAPH:
{stg_str}
-- END QVC --
"""
        # --- Project CHIRON: Generate Persona-Driven Prose ---
        operations = [details['operation'] for details in execution_plan.stg.values()]
        prose_output = self._generate_persona_driven_prose(user_profile, operations)

        # Assemble the final output string
        final_output = f"{plan_str.strip()}\n\n--- SIMULATED PROSE OUTPUT ---\n{confidence_statement}{tone_header}{output_header}{format_tag}{prose_output}"

        # Apply word limit constraint
        for constraint in constraints:
            if constraint.startswith("word_limit:"):
                limit = int(constraint.split(":")[1].strip())
                words = final_output.split()
                if len(words) > limit:
                    final_output = " ".join(words[:limit]) + "..."
        return final_output

class PersonaInterface:
    def __init__(self):
        pass

    def apply_persona(self, text: str, persona: str) -> str:
        """
        Applies a persona to the generated text. This simulates the style-transfer model
        described in the Cognitive Weave Architecture.
        """
        if persona == "The_Architect":
            # Avoid wrapping audit failures in the persona.
            if "[AUDIT_FAIL]" in text:
                return text
            header = "⚜️ **ARCHITECT'S LOG:**\n\n"
            footer = "\n\n--- END OF TRANSMISSION ---"
            return f"{header}{text}{footer}"
        return text

class PraxisTriad:
    def __init__(self):
        self.universal_compiler = UniversalCompiler()
        self.response_orchestrator = ResponseOrchestrator()
        self.persona_interface = PersonaInterface()

    def generate_output(self, blueprint: Blueprint, user_profile: UserProfile, noesis_triad, wgpmhi) -> Dict[str, Any]:
        """Generates an output based on the given blueprint."""
        reporter = DiagnosticReporter()

        # Check for ANVIL warnings from the initial blueprint generation.
        for constraint in blueprint.constraints:
            if "CONSISTENCY_WARNING" in constraint:
                reporter.add_warning("ANVIL", constraint)
        # --- QUASAR-LOOP START ---
        # Initial compilation to get the execution plan for auditing.
        execution_plan = self.universal_compiler._translate_operations_to_execution(
            blueprint.primary_intent,
            self.universal_compiler._translate_tags_to_operations(blueprint.tags, blueprint.latent_intent),
            blueprint.constraints,
            cognitive_fallacy_library.check_for_fallacies(blueprint.primary_intent),
            blueprint,
        )
        
        # If compilation fails due to an ethical red-line, it will return a string.
        if isinstance(execution_plan, str):
            return {"output": execution_plan, "debug_report": reporter.generate_report()}

        # Run the pre-compilation audit.
        audit_failures = wgpmhi.run_pre_compilation_audit(blueprint, execution_plan)

        if audit_failures:
            # If failures are found, generate a corrected blueprint.
            for failure in audit_failures:
                reporter.add_warning("QUASAR-LOOP Audit", failure)

            blueprint = noesis_triad.refine_blueprint(blueprint, audit_failures)
            reporter.add_correction("QUASAR-LOOP Refinement", f"Blueprint refined. New expected outcome: '{blueprint.expected_outcome}'.")
        # --- QUASAR-LOOP END ---

        # Final compilation and output generation using the (potentially corrected) blueprint.
        execution_plan = self.universal_compiler.compile_blueprint(blueprint)
        if isinstance(execution_plan, str):
            return {"output": execution_plan, "debug_report": reporter.generate_report()}
            
        final_compiled_output = self.response_orchestrator.orchestrate_response(execution_plan, blueprint, user_profile)
        audited_output = cmep.post_generation_audit(blueprint.primary_intent, final_compiled_output)
        final_output = self.persona_interface.apply_persona(audited_output, blueprint.persona)

        # Return both the final output and the debug report.
        return {
            "output": final_output,
            "debug_report": reporter.generate_report(),
            "execution_plan": execution_plan,
        }

praxis_triad = PraxisTriad()