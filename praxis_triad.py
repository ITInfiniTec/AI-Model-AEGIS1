# praxis_triad.py

import re
from typing import List, Dict, Any
from data_structures import Blueprint
from cmep import cmep
from cognitive_fallacy_library import cognitive_fallacy_library
from diagnostic_reporter import DiagnosticReporter
from prometheus_iop import prometheus_iop

class UniversalCompiler:
    def __init__(self):
        pass

    def _translate_tags_to_operations(self, tags: List[Dict[str, str]], latent_intent: str) -> List[str]:
        """Tier 2 Simulation: Translates tags into a sequence of logical operations."""
        operations = []
        tag_values = {tag['value'] for tag in tags}

        # A simple mapping from keywords to pseudo-operations
        # Latent intent can modify or add to the plan
        if "strategic framework" in latent_intent:
            operations.append("OP_GENERATE_STRATEGIC_FRAMEWORK")
        elif "summarize" in tag_values:
            operations.append("OP_TEXT_SUMMARIZE")

        if 'summarize' in tag_values:
            operations.append("OP_TEXT_SUMMARIZE")
        if 'quantum' in tag_values or 'physics' in tag_values:
            operations.append("OP_FETCH_KNOWLEDGE(topic='quantum_physics')")
        if 'blockchain' in tag_values:
            operations.append("OP_FETCH_KNOWLEDGE(topic='blockchain')")
        if any(ai_tag in tag_values for ai_tag in ['ai', 'ml', 'ann', 'gnn']):
            operations.append("OP_FETCH_KNOWLEDGE(topic='ai_ml')")
        if any(creative_tag in tag_values for creative_tag in ['poem', 'story', 'imagine', 'create']):
            operations.append("OP_CREATIVE_WRITING")
        
        # --- Project CHRONOS: New Tag Recognition ---
        if any(tag.get("type") == "MODEL_PROTOCOL" and tag.get("value") == "PREDICTIVE_MODEL: REQUIRED" for tag in tags):
            operations.append("OP_FETCH_TIME_SERIES_DATA")
            operations.append("OP_ANALYZE_SERIES(model='ARIMA_SIM')")
            operations.append("OP_GENERATE_FORECAST")

        if not operations:
            operations.append("OP_GENERAL_QUERY")
        
        return operations

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
            elif op == "OP_GENERATE_FORECAST":
                # Forecasting depends on analysis.
                analysis_task = next((tid for tid, details in stg.items() if details["operation"].startswith("OP_ANALYZE_SERIES")), None)
                stg[task_id] = {"operation": op, "depends_on": [analysis_task] if analysis_task else []}
            else:
                stg[task_id] = {"operation": op, "depends_on": []}
            task_counter += 1
        return stg

    def _translate_operations_to_execution(self, intent: str, operations: List[str], constraints: List[str], fallacies: List[str]) -> str:
        """Tier 3 Simulation: Compiles operations into a final execution string (pseudo-QVC). Uses primary_intent for hash."""
        constraints_str = ", ".join(constraints) if constraints else "None"
        ops_str = "\n    ".join(operations)
        fallacy_warnings = ""
        if fallacies:
            fallacy_str = ", ".join(fallacies)
            fallacy_warnings = f"\nWARNING: FALLACY_DETECTED({fallacy_str}). Proceeding with caution."

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
        stg_str = "\n".join([f"  {task_id}: {details}" for task_id, details in stg.items()])

        # --- Constraint Processing for QVC Execution ---
        format_constraint = next((c.split('(')[1].strip(')') for c in constraints if c.startswith("FORMAT")), "TEXT_BLOCK")
        audience_constraint = next((c.split('(')[1].strip(')') for c in constraints if c.startswith("AUDIENCE")), "GENERAL_USER")

        # --- CASSANDRA Tag Processing for QVC Execution ---
        tags = {t['value'] for t in blueprint.tags}
        external_data_required = "YES" if "REQUIRE_EXTERNAL_DATA" in tags else "NO"
        safety_priority = "HIGH" if "SAFETY_PRIORITY" in tags else "STANDARD"
        ethical_consult_required = "YES" if "ETHICAL_CONSULT" in tags else "NO"

        # Simulate the "Vector Cryptographic QVC (unfolding execution)"
        execution_string = f"""
-- BEGIN QVC UNFOLDING EXECUTION --
INTENT_HASH: {hash(intent)} # In a real system, this would be a cryptographic hash.
CONSTRAINTS: [{constraints_str}]
TARGET_FORMAT: {format_constraint}
TARGET_AUDIENCE: {audience_constraint}{fallacy_warnings}
EXTERNAL_DATA_REQUIRED: {external_data_required}
SAFETY_PRIORITY_LEVEL: {safety_priority}
ETHICAL_CONSULTATION: {ethical_consult_required}

SIMULATED_FORECAST_RESULT: {forecast}
SEQUENTIAL_TASK_GRAPH:
{stg_str}

-- END QVC --
"""
        return execution_string.strip()

    def compile_blueprint(self, blueprint: Blueprint, primary_intent: str, tags: List[Dict[str, str]], latent_intent: str, constraints: List[str]) -> str:
        """Compiles the blueprint into an executable output."""
        # Enforce ethical considerations (from Conceptual Audit in NoesisTriad)
        if "violation detected" in blueprint.ethical_considerations:
            return "Ethical red-line violation detected. Output cannot be generated."

        # Self-Correcting Compiler Protocol: Check for cognitive fallacies.
        detected_fallacies = cognitive_fallacy_library.check_for_fallacies(blueprint.primary_intent)

        # Tier 2: Tag-to-Operation Translation
        operations = self._translate_tags_to_operations(tags, latent_intent)

        # Tier 3: Operation-to-Execution Translation (pass full blueprint for tag access)
        execution_plan = self._translate_operations_to_execution(primary_intent, operations, constraints, detected_fallacies)

        # The compiler's sole job is to produce the execution plan.
        return execution_plan

class ResponseOrchestrator:
    """Applies final formatting, constraints, and confidence layers to the output."""
    def _generate_persona_driven_prose(self, user_profile: UserProfile, operations: List[str]) -> str:
        """Generates simulated prose using analogies based on user passions."""
        passions = user_profile.passions
        prose = "Based on the execution plan, here is a summary of the requested topics. "

        if "OP_FETCH_KNOWLEDGE(topic='blockchain')" in operations:
            if "chess" in passions:
                prose += "Blockchain acts like a grandmaster's logbook, where every move (transaction) is recorded immutably for all to see. "
            else:
                prose += "Blockchain is a decentralized, distributed ledger technology. "

        prose += "Artificial Intelligence (AI) is a wide-ranging branch of computer science concerned with building smart machines capable of performing tasks that typically require human intelligence. These fields are complex and have many nuances."
        return prose

    def orchestrate_response(self, execution_plan: str, blueprint: Blueprint, user_profile: UserProfile) -> str:
        """Constructs the final output string from the raw execution results and blueprint."""
        constraints = blueprint.constraints
        expected_outcome = blueprint.expected_outcome

        # Confidence Layer Integration (Project VESTA)
        confidence_tag = next((tag for tag in blueprint.tags if tag.get("type") == "CONTEXT_CONFIDENCE"), None)
        confidence_statement = ""
        if confidence_tag and confidence_tag.get("value") == "LOW":
            confidence_statement = "CONFIDENCE_NOTE: Context confidence is low. External verification is recommended.\n"

        # Audience Tone Modulation
        audience_constraint = next((c.split('(')[1].strip(')') for c in constraints if c.startswith("AUDIENCE")), "GENERAL_USER")
        tone_header = ""
        if audience_constraint in ["beginner", "novice"]:
            tone_header = f"SIMULATED_TONE: (Simplified for {audience_constraint})\n"
        elif audience_constraint in ["expert", "professional"]:
            tone_header = f"SIMULATED_TONE: (Technical prose for {audience_constraint})\n"

        # Format Application
        format_constraint = next((c.split('(')[1].strip(')') for c in constraints if c.startswith("FORMAT")), "TEXT_BLOCK")
        format_tag = ""
        if format_constraint != "TEXT_BLOCK":
             format_tag = f"SIMULATED_FORMAT_APPLIED({format_constraint})\n"

        # Expected Outcome Formatting
        output_header = ""
        if expected_outcome == "brief summary":
            output_header = "Here is a brief summary as requested:\n"

        # --- Project CHIRON: Generate Persona-Driven Prose ---
        prose_output = self._generate_persona_driven_prose(user_profile, re.findall(r"'operation': '(.*?)'", execution_plan))

        # Assemble the final output string
        final_output = f"{execution_plan}\n\n--- SIMULATED PROSE OUTPUT ---\n{confidence_statement}{tone_header}{output_header}{format_tag}{prose_output}"

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
        )

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
        execution_plan = self.universal_compiler.compile_blueprint(
            blueprint,
            blueprint.primary_intent,
            blueprint.tags,
            blueprint.latent_intent,
            blueprint.constraints
        )
        final_compiled_output = self.response_orchestrator.orchestrate_response(execution_plan, blueprint, user_profile)
        audited_output = cmep.post_generation_audit(blueprint.primary_intent, final_compiled_output)
        final_output = self.persona_interface.apply_persona(audited_output, blueprint.persona)

        # Return both the final output and the debug report.
        return {"output": final_output, "debug_report": reporter.generate_report()}

praxis_triad = PraxisTriad()