# aegis_core.py

from noesis_triad import noesis_triad
from praxis_triad import PraxisTriad
from wgpmhi import wgpmhi
from data_structures import UserProfile, MemoryNode
from cognitive_packet_generator import cognitive_packet_generator
from isp_protocol import isp
from prometheus_iop import prometheus_iop
from state_manager import state_manager
import json
from logger import log
import sys

class AEGIS_Core:
    """
    The core abstraction layer for the AEGIS engine. Encapsulates the entire
    cognitive pipeline into a single, manageable component.
    """
    def __init__(self, user_id: str, user_profile: UserProfile):
        # Instantiate all core components of the AEGIS architecture.
        # @RISK: user_id and user_profile are assumed to be validated before this point.
        self.user_id = user_id
        self.user_profile = user_profile
        self.noesis_triad = noesis_triad # This is a singleton instance
        self.praxis_triad = PraxisTriad() # Instantiate PraxisTriad
        self.wgpmhi = wgpmhi
        self.isp = isp
        self.prometheus_iop = prometheus_iop

    def process_prompt(self, prompt: str) -> dict:
        """
        Processes a user prompt through the entire AEGIS lifecycle and returns
        a dictionary containing all generated artifacts.
        """
        try:
            # Ensure the current user's profile is available to the Noesis Triad.
            self.noesis_triad.context_synthesizer.user_profiles[self.user_id] = self.user_profile
    
            # Generate a blueprint using the Noesis Triad
            blueprint = self.noesis_triad.generate_blueprint(self.user_id, prompt)
    
            # Generate an output using the Praxis Triad
            generation_result = self.praxis_triad.generate_output(blueprint, self.user_profile, self.noesis_triad, self.wgpmhi)
            output = generation_result["output"]
            debug_report = generation_result["debug_report"]
            execution_plan = generation_result["execution_plan"]
    
            # Run the WGPMHI tests to get the results needed for the Cognitive Packet.
            # Pass None for packet initially as it hasn't been generated yet.
            # Renaming the key for clarity as requested.
            wgpmhi_results = self.wgpmhi.run_tests(self.user_profile, blueprint, execution_plan, output, self.noesis_triad, None)
            wgpmhi_results['time_series_planning_check'] = wgpmhi_results.pop('predictive_workflow_check', 'N/A')
    
            # Generate a Cognitive Packet for training.
            cognitive_packet = cognitive_packet_generator.generate_packet(blueprint, output, wgpmhi_results, debug_report)
    
            # Store the Cognitive Packet in Long-Term Memory.
            self.noesis_triad.context_synthesizer.update_long_term_memory(self.user_id, cognitive_packet)
    
            # --- Project PROMETHEUS: Final Output and External Audit ---
            # 1. Simulate external audit via ISP (existing Project PANDORA)
            external_request = {"packet_id": cognitive_packet.packet_id}
            isp_response = self.isp.handle_external_audit_request(external_request)
    
            # 2. Simulate sending the Cognitive Packet to the Prometheus monitoring system (NEW)
            prometheus_queue_status = self.prometheus_iop.send_cognitive_packet(cognitive_packet)
            # Return all generated artifacts in a structured dictionary.
            return {
                "final_output": output,
                "blueprint": blueprint.model_dump(),
                "wgpmhi_results": wgpmhi_results,
                "orthrus_debug_report": str(debug_report),
                "cognitive_packet": cognitive_packet.model_dump(mode='json'),
                "isp_audit_response": isp_response,
                "prometheus_queue_status": prometheus_queue_status,
            }
        except (ValueError, TypeError) as e:
            # Catch specific data or type-related errors.
            log.error(f"Data processing error for user '{self.user_id}': {e}", exc_info=True)
            error_message = f"DATA_PROCESSING_ERROR: An error occurred processing the request. Details: {e}"
            return self._generate_error_response(error_message, type(e).__name__)
        except Exception as e:
            # Catch any unhandled exception during the process and return a structured error.
            log.critical(f"Unhandled critical error for user '{self.user_id}': {e}", exc_info=True)
            error_message = f"CRITICAL_ERROR: An unhandled exception occurred: {e}"
            return self._generate_error_response(error_message, type(e).__name__)

    # --- Command Handler Methods ---
    def get_memory(self):
        """Retrieves memory for the current user."""
        return state_manager.get_memory_for_user(self.user_id)

    def get_memory_node(self, node_id: str) -> MemoryNode | None:
        """Retrieves a specific memory node for the current user."""
        for node in self.get_memory():
            if node.node_id == node_id:
                return node
        return None

    def clear_memory(self):
        """Clears memory for the current user."""
        state_manager.clear_memory(self.user_id)

    def get_config(self):
        """Retrieves the full system configuration."""
        from config_loader import config_loader
        return config_loader.get_full_config()

    def _generate_error_response(self, message: str, error_type: str) -> dict:
        """Creates a standardized error response dictionary."""
        return {
            "final_output": message,
            "blueprint": {},
            "wgpmhi_results": {"system_stability": f"Fail: {error_type}"},
            "orthrus_debug_report": message,
            "cognitive_packet": {},
            "isp_audit_response": {"status": "error", "message": "Processing failed."},
            "prometheus_queue_status": {"status": "error", "message": "Packet not generated."},
        }

class CommandHandler:
    """Handles the registration and execution of CLI commands for the AEGIS engine."""
    def __init__(self, aegis_engine: AEGIS_Core):
        self.aegis_engine = aegis_engine
        self._commands = {
            "exit": self.exit_session,
            "quit": self.exit_session,
            "view_memory": self.view_memory,
            "clear_memory": self.clear_memory,
            "view_node": self.view_node,
            "view_config": self.view_config,
            "stress_test": self.run_stress_test,
        }

    def is_command(self, name: str) -> bool:
        """Checks if a given name corresponds to a registered command."""
        return name in self._commands

    def execute(self, name: str, *args: str):
        """Executes a registered command with the given arguments."""
        try:
            self._commands[name](*args)
        except TypeError:
            print(f"Error: Invalid arguments for command '{name}'. Please check usage.")
        except Exception as e:
            print(f"An error occurred executing command '{name}': {e}")

    def exit_session(self, *args):
        """Terminates the AEGIS Core session."""
        print("Terminating session. AEGIS Core shutting down.")
        sys.exit(0)

    def view_memory(self, *args):
        """Displays a summary of the long-term memory."""
        print(f"\n--- LONG-TERM MEMORY (USER: {self.aegis_engine.user_id}) ---")
        memory = self.aegis_engine.get_memory()
        if not memory:
            print("No memories found.")
        else:
            for i, node in enumerate(memory):
                packet = node.packet_reference
                wgpmhi_results = packet.wgpmhi_results
                pass_count = sum(1 for result in wgpmhi_results.values() if "Pass" in str(result))
                total_tests = len(wgpmhi_results) - 1 # Exclude anti_fragility_protocol_status

                print(f"  Memory Node {i+1}:")
                print(f"    - Node ID: {node.node_id}")
                print(f"    - Timestamp: {node.timestamp.isoformat()}")
                print(f"    - Metrics: Perf={node.performance_score:.2f} | Risk={packet.risk_score:.2f} | Novelty={packet.novelty_score:.2f}")
                print(f"    - Audit: {pass_count}/{total_tests} Tests Passed")
                print(f"    - Prompt: '{packet.intent['primary'][:70]}...'")

    def clear_memory(self, *args):
        """Clears the long-term memory for the current session."""
        self.aegis_engine.clear_memory()
        print("Long-term memory for the current session has been cleared.")

    def view_node(self, *args):
        """Displays the full details of a specific memory node."""
        if not args:
            print("Usage: view_node <node_id>")
            return
        node_id = args[0]
        node = self.aegis_engine.get_memory_node(node_id)
        if node:
            print(f"\n--- DETAILS FOR MEMORY NODE: {node_id} ---")
            print(node.model_dump_json(indent=4))
        else:
            print(f"Memory Node with ID '{node_id}' not found in the current session.")

    def view_config(self, *args):
        """Displays the current system configuration."""
        print("\n--- CURRENT AEGIS CORE CONFIGURATION ---")
        config = self.aegis_engine.get_config()
        print(json.dumps(config, indent=4))

    def run_stress_test(self, *args):
        """Runs the V-Architect integration stress test."""
        from v_architect_sim import VArchitectSimulator
        print("Initiating V-Architect stress test from CLI...")
        simulator = VArchitectSimulator()
        simulator.run_stress_test()