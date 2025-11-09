# praxis_triad.py

import re
from typing import List, Dict, Any
from data_structures import Blueprint, UserProfile, SemanticTag, ExecutionPlan
from cmep import cmep
from cognitive_fallacy_library import cognitive_fallacy_library
from diagnostic_reporter import DiagnosticReporter
from config_loader import config_loader
from prometheus_iop import prometheus_iop

class UniversalCompiler:
    def __init__(self):
        # Load the compiler rules from the central configuration.
        self.rules = config_loader.get_compiler_rules()

    def _translate_tags_to_operations(self, tags: List[SemanticTag], latent_intent: str) -> List[str]:
        """
        Translates tags into a sequence of logical operations using a configurable rule-based engine.
        This is more scalable and knowable than a hardcoded if/else block.
        """
        operations = set()
        tag_values = {tag.value for tag in tags}

        for rule in self.rules:
            conditions = rule.get("conditions", {})
            
            # Condition: Check if any of a list of tag values are present.
            if "tags_include_any" in conditions and not any(tag in tag_values for tag in conditions["tags_include_any"]):
                continue

            # Condition: Check for a specific latent intent string.
            if "latent_intent_is" in conditions and latent_intent != conditions["latent_intent_is"]:
                continue

            # Condition: Check for a specific tag with a given type and value.
            if "tag_is" in conditions:
                required_tag = conditions["tag_is"]
                if not any(tag.type == required_tag.get("type") and tag.value == required_tag.get("value") for tag in tags):
                    continue

            # If all conditions passed, add the operation(s).
            if "operation" in rule: operations.add(rule["operation"])
            if "operations" in rule: operations.update(rule["operations"])

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
        word_limit_constraint = next((c for c in constraints if c.startswith("word_limit:")), None)
        word_limit = None
        if word_limit_constraint:
            try:
                word_limit = int(word_limit_constraint.split(":")[1].strip())
            except (ValueError, IndexError):
                word_limit = None # Ignore malformed constraint

        # --- CASSANDRA Tag Processing for QVC Execution ---
        tag_values = {t.value for t in blueprint.tags}
        
        return ExecutionPlan(
            intent_hash=hash(intent),
            constraints=constraints,
            target_format=format_constraint,
            target_audience=audience_constraint,
            fallacy_warnings=fallacies,
            word_limit=word_limit,
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
    def __init__(self):
        # Load dependency rules from the central configuration.
        rules = config_loader.get_stg_dependency_rules()
        self.op_types = rules.get("operation_types", {})
        self.dependencies = rules.get("dependencies", {})

    def _get_op_type(self, operation: str) -> str | None:
        """Helper to find the type of a given operation string."""
        for op_type, op_list in self.op_types.items():
            if any(op_name in operation for op_name in op_list):
                return op_type
        return None

    def generate_stg(self, operations: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Generates a Sequential Task Graph from a list of operations using a
        configurable, rule-based dependency engine.
        """
        stg = {}
        tasks_by_type: Dict[str, List[str]] = {op_type: [] for op_type in self.op_types}

        # First pass: Create all tasks and categorize them by type.
        for i, op in enumerate(operations):
            task_id = f"TASK_{i+1}"
            op_type = self._get_op_type(op)
            stg[task_id] = {"operation": op, "depends_on": []}
            if op_type:
                tasks_by_type[op_type].append(task_id)
            # Add failure paths for critical fetch operations.
            if op_type == "knowledge_fetch":
                stg[task_id]["on_failure"] = f"LOG_ERROR_AND_HALT({task_id})"

        # Second pass: Wire dependencies based on the rules.
        for task_id, details in stg.items():
            op_type = self._get_op_type(details["operation"])
            if op_type and op_type in self.dependencies:
                for required_dependency_type in self.dependencies[op_type]:
                    details["depends_on"].extend(tasks_by_type.get(required_dependency_type, []))

        return stg

class ResponseOrchestrator:
    """Applies final formatting, constraints, and confidence layers to the output."""
    def __init__(self):
        # Project CHIRON: A dictionary of analogies to connect topics to user passions.
        self.persona_interface = PersonaInterface()
        # Load conversational configurations once during initialization.
        self.conv_config = config_loader.get_conversational_config()
        self.greetings = set(self.conv_config.get("greetings", []))
        self.short_interactions = self.conv_config.get("short_interactions", {})
        self.passion_analogies = config_loader.get_passion_analogies()
        self.analogy_fallback = self.passion_analogies.pop("fallback_template", "The topic of {topic} is complex.")
        self.op_topic_mapping = config_loader.get_operation_topic_mapping()

    def _generate_persona_driven_prose(self, user_profile: UserProfile, operations: List[str], prompt: str) -> str:
        """Generates simulated prose using analogies based on user passions."""
        passions = user_profile.passions
        prose_segments = ["Based on the execution plan, here is a summary of the requested topics."]
        lower_prompt = prompt.lower().strip()

        # --- Conversational Reply Mapping ---
        # Handle common, short interactions with appropriate, direct replies.
        if lower_prompt in self.short_interactions:
            return self.short_interactions[lower_prompt]

        # If it's a general query, check if it's a recognized greeting.
        if len(operations) == 1 and operations[0] == "OP_GENERAL_QUERY":
            if lower_prompt in self.greetings:
                return self.conv_config.get("greeting_response", "Hello.")
            else:
                # For other general queries, provide a neutral analysis summary.
                return self.conv_config.get("neutral_query_response", "Query analyzed.")

        # Map all operations to topics more robustly using the configuration.
        topics_in_plan = set()
        for op in operations:
            for op_name, topics in self.op_topic_mapping.items():
                if op.startswith(op_name):
                    if topics == "extract_from_op":
                        match = re.search(r"\('([^']*)'\)", op)
                        if match: topics_in_plan.add(match.group(1))
                    else:
                        topics_in_plan.update(topics)
                    break

        for topic in sorted(list(topics_in_plan)): # Sort for consistent output
            analogy_found = False
            for passion in passions:
                if topic in self.passion_analogies and passion in self.passion_analogies[topic]:
                    prose_segments.append(self.passion_analogies[topic][passion])
                    analogy_found = True
                    break  # Use the first matching passion-analogy
            if not analogy_found:
                # Provide a generic fallback if no specific analogy is found for the topic
                prose_segments.append(self.analogy_fallback.format(topic=topic.replace('_', ' ')))

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
        prose_output = self._generate_persona_driven_prose(user_profile, operations, blueprint.primary_intent)

        # --- Intelligent Persona Application ---
        # Check if the prose is a simple conversational reply. If so, return it directly.
        # Otherwise, wrap it in the full persona.
        if self.persona_interface.is_conversational(prose_output):
            return prose_output
        else:
            # Assemble the final output string
            final_output = f"{plan_str.strip()}\n\n--- SIMULATED PROSE OUTPUT ---\n{confidence_statement}{tone_header}{output_header}{format_tag}{prose_output}"

            # Apply word limit constraint
            if execution_plan.word_limit is not None:
                words = final_output.split()
                if len(words) > execution_plan.word_limit:
                    final_output = " ".join(words[:execution_plan.word_limit]) + "..."
            return self.persona_interface.apply_persona(final_output, blueprint.persona)

class PersonaInterface:
    def __init__(self):
        # Load all possible conversational replies for efficient checking.
        self.conversational_replies = config_loader.get_conversational_config().get("all_replies", [])
        # Load all persona definitions from the central configuration.
        self.personas = config_loader.get_personas_config()

    def apply_persona(self, text: str, persona: str) -> str:
        """
        Applies a persona to the generated text. This simulates the style-transfer model
        described in the Cognitive Weave Architecture. It now intelligently avoids
        wrapping simple, conversational replies and uses a data-driven persona definition.
        """
        persona_config = self.personas.get(persona)
        if persona_config:
            if "[AUDIT_FAIL]" in text: return text # Avoid wrapping audit failures.

            header = persona_config.get("header", "")
            footer = persona_config.get("footer", "")
            return f"{header}{text}{footer}"
        return text

    def is_conversational(self, text: str) -> bool:
        """Checks if a given text is a known simple conversational reply."""
        return text in self.conversational_replies


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
        final_output = audited_output # Persona is now applied inside the orchestrator

        # Return both the final output and the debug report.
        return {
            "output": final_output,
            "debug_report": reporter.generate_report(),
            "execution_plan": execution_plan,
        }

praxis_triad = PraxisTriad()