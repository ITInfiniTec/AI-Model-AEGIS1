# aegis_core.py

from noesis_triad import noesis_triad
from data_integrity_protocol import data_integrity_protocol
from praxis_triad import PraxisTriad
from wgpmhi import wgpmhi
from data_structures import UserProfile, MemoryNode
from cognitive_packet_generator import cognitive_packet_generator
from isp_protocol import isp
from prometheus_iop import prometheus_iop
from state_manager import state_manager
import json
from logger import log
from datetime import datetime
import sys

class AEGIS_Core:
    """
    The core abstraction layer for the AEGIS engine. Encapsulates the entire
    cognitive pipeline into a single, manageable component.
    """
    def __init__(self, user_id: str, user_profile: UserProfile):
        # Validate the incoming user profile at the boundary to ensure system integrity.
        self.user_profile = data_integrity_protocol.validate_user_profile(user_profile)
        self.user_id = self.user_profile.user_id
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
            external_request = {"packet_id": cognitive_packet.packet_id} # type: ignore
            isp_response = self.isp.handle_external_audit_request(external_request, self.noesis_triad)
    
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

def command(*aliases):
    """A decorator to register a method as a CLI command with optional aliases."""
    def decorator(func):
        func._is_command = True
        func._aliases = aliases
        return func
    return decorator

class CommandHandler:
    """Handles the registration and execution of CLI commands for the AEGIS engine."""
    def __init__(self, aegis_engine: AEGIS_Core):
        self.aegis_engine = aegis_engine
        self._commands = {}
        # Use introspection to dynamically discover and register commands.
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_is_command'):
                # The primary command name is the function name
                self._commands[attr.__name__] = attr
                # Add any aliases defined in the decorator
                for alias in attr._aliases:
                    self._commands[alias] = attr

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

    @command("quit")
    def exit_session(self, *args):
        """Terminates the AEGIS Core session."""
        print("Terminating session. AEGIS Core shutting down.")
        sys.exit(0)

    @command()
    def view_memory(self, *args):
        """
        Displays a paginated summary of the long-term memory.
        Usage: view_memory [page] [size] [start:YYYY-MM-DD] [end:YYYY-MM-DD] [--summary]
        """
        page_number = 1
        page_size = 5
        start_date = None
        end_date = None
        summary_view = False
        positional_args = []

        try:
            for arg in args:
                if arg.startswith("start:"):
                    start_date = datetime.fromisoformat(arg.split(":")[1])
                elif arg.startswith("end:"):
                    # Add one day to the end date to make the range inclusive
                    from datetime import timedelta
                    end_date = datetime.fromisoformat(arg.split(":")[1]) + timedelta(days=1) # type: ignore
                elif arg == "--summary":
                    summary_view = True
                else:
                    positional_args.append(arg)
            
            if positional_args:
                page_number = int(positional_args[0])
            if len(positional_args) > 1:
                page_size = int(positional_args[1])

        except (ValueError, IndexError) as e:
            print(f"Error parsing arguments: {e}")
            print("Usage: view_memory [page] [size] [start:YYYY-MM-DD] [end:YYYY-MM-DD] [--summary]")
            return

        print(f"\n--- LONG-TERM MEMORY (USER: {self.aegis_engine.user_id}) ---")
        memory = self.aegis_engine.get_memory()

        # Apply date filtering before pagination
        filtered_memory = [
            node for node in memory
            if (not start_date or node.timestamp >= start_date) and \
               (not end_date or node.timestamp < end_date)
        ]

        if not filtered_memory:
            print("No memories found.")
            return

        total_memories = len(filtered_memory)
        total_pages = (total_memories + page_size - 1) // page_size

        if not 1 <= page_number <= total_pages:
            print(f"Error: Invalid page number. Please enter a number between 1 and {total_pages}.")
            return

        from datetime import timedelta
        filter_str = f" | Filters: start={start_date.date() if start_date else 'N/A'}, end={(end_date - timedelta(days=1)).date() if end_date else 'N/A'}"
        print(f"Displaying page {page_number} of {total_pages} ({total_memories} total memories){filter_str}\n")

        start_index = (page_number - 1) * page_size
        end_index = start_index + page_size
        paginated_memory = filtered_memory[start_index:end_index]

        for i, node in enumerate(paginated_memory, start=start_index):
            packet = node.packet_reference
            if summary_view:
                print(f"  Node {i+1:<3} [{node.timestamp.date()}] Perf:{node.performance_score:.2f} | Prompt: '{packet.intent['primary'][:50]}...'")
            else:
                print(f"  Memory Node {i+1}:")
                print(f"    - Node ID: {node.node_id}")
                print(f"    - Timestamp: {node.timestamp.isoformat()}")
                print(f"    - Metrics: Perf={node.performance_score:.2f} | Risk={packet.risk_score:.2f} | Novelty={packet.novelty_score:.2f}")
                print(f"    - Prompt: '{packet.intent['primary'][:70]}...'")

    @command()
    def clear_memory(self, *args):
        """Clears the long-term memory for the current session."""
        print("WARNING: This will permanently delete all memory for the current session.")
        confirmation = input("Are you sure you want to proceed? (yes/no): ").lower()

        if confirmation == 'yes':
            self.aegis_engine.clear_memory()
            print("Long-term memory for the current session has been cleared.")
        else:
            print("Operation aborted. Memory was not cleared.")

    @command()
    def view_node(self, *args):
        """Displays the full details of a specific memory node."""
        if not args:
            print("Usage: view_node <node_id>")
            return
        node_id = args[0]
        node = self.aegis_engine.get_memory_node(node_id)
        if node:
            print(f"\n--- DETAILS FOR MEMORY NODE: {node_id} ---")
            # Instead of a raw JSON dump, print a structured, readable summary.
            packet = node.packet_reference
            print(f"  - Timestamp: {node.timestamp.isoformat()}")
            print(f"  - Performance Score: {node.performance_score:.4f}")
            print("\n  --- INTENT ---")
            print(f"  - Primary: {packet.intent['primary']}")
            print(f"  - Latent: {packet.intent['latent']}")
            print("\n  --- METRICS ---")
            print(f"  - Risk Score: {packet.risk_score:.4f}")
            print(f"  - Novelty Score: {packet.novelty_score:.4f}")
            print("\n  --- PROSE OUTPUT ---")
            print(f"  {packet.output_summary}")
            print("\n  --- WGPMHI AUDIT RESULTS ---")
            for test, result in packet.wgpmhi_results.items():
                print(f"    - {test}: {result}")
            print("\n  --- DEBUG REPORT ---")
            print(f"  {packet.debug_report}")
        else:
            print(f"Memory Node with ID '{node_id}' not found in the current session.")

    @command()
    def view_config(self, *args):
        """
        Displays the current system configuration, or a specific section.
        Usage: view_config [section_key]
        """
        print("\n--- CURRENT AEGIS CORE CONFIGURATION ---")
        config = self.aegis_engine.get_config()

        if not args:
            # If no section is specified, print the entire configuration.
            print(json.dumps(config, indent=4))
            return

        section_key = args[0]
        section_data = config.get(section_key)

        if section_data is not None:
            print(json.dumps(section_data, indent=4))
        else:
            print(f"\nError: Configuration section '{section_key}' not found.")
            print(f"Available sections are: {', '.join(config.keys())}")

    @command()
    def run_stress_test(self, *args):
        """
        Runs the V-Architect integration stress test.
        Usage: stress_test [cycles]
        """
        from v_architect_sim import VArchitectSimulator
        try:
            cycles = int(args[0]) if args else 3 # Default to 3 cycles if not specified
        except (ValueError, IndexError):
            print("Usage: stress_test [cycles]")
            return
        print("Initiating V-Architect stress test from CLI...")
        simulator = VArchitectSimulator()
        simulator.run_stress_test(cycles)