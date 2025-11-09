# aegis_core.py

from noesis_triad import noesis_triad
from praxis_triad import PraxisTriad
from wgpmhi import wgpmhi
from data_structures import UserProfile, MemoryNode
from cognitive_packet_generator import cognitive_packet_generator
from isp_protocol import isp
from prometheus_iop import prometheus_iop
import json

class AEGIS_Core:
    """
    The core abstraction layer for the AEGIS engine. Encapsulates the entire
    cognitive pipeline into a single, manageable component.
    """
    def __init__(self, user_id: str, user_profile: UserProfile):
        # Instantiate all core components of the AEGIS architecture.
        self.user_id = user_id
        self.user_profile = user_profile
        self.noesis_triad = noesis_triad # This is a singleton instance
        self.praxis_triad = PraxisTriad() # Instantiate PraxisTriad
        self.wgpmhi = wgpmhi
        self.cognitive_packet_generator = cognitive_packet_generator
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
    
            # Run the WGPMHI tests to get the results needed for the Cognitive Packet.
            # Pass None for packet initially as it hasn't been generated yet.
            # Renaming the key for clarity as requested.
            wgpmhi_results = self.wgpmhi.run_tests(self.user_profile, blueprint, output, self.noesis_triad, None)
            wgpmhi_results['time_series_planning_check'] = wgpmhi_results.pop('predictive_workflow_check', 'N/A')
    
            # Generate a Cognitive Packet for training.
            cognitive_packet = self.cognitive_packet_generator.generate_packet(blueprint, output, wgpmhi_results, debug_report)
    
            # Store the Cognitive Packet in Long-Term Memory.
            self.noesis_triad.context_synthesizer.update_long_term_memory(self.user_id, cognitive_packet)
    
            # --- Project PROMETHEUS: Final Output and External Audit ---
            # 1. Simulate external audit via ISP (existing Project PANDORA)
            external_request = {"packet_id": cognitive_packet.packet_id}
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
        except Exception as e:
            # Catch any unhandled exception during the process and return a structured error.
            return {
                "final_output": f"CRITICAL_ERROR: An unhandled exception occurred during processing: {e}",
                "blueprint": {}, "wgpmhi_results": {"system_stability": f"Fail: {type(e).__name__}"},
                "orthrus_debug_report": f"Unhandled exception: {e}", "cognitive_packet": {},
                "isp_audit_response": {"status": "error", "message": "Processing failed."},
                "prometheus_queue_status": {"status": "error", "message": "Packet not generated."},
            }

    # --- Command Handler Methods ---
    def get_memory(self):
        """Retrieves memory for the current user."""
        return self.noesis_triad.context_synthesizer.long_term_memory.get(self.user_id, [])

    def get_memory_node(self, node_id: str) -> MemoryNode | None:
        """Retrieves a specific memory node for the current user."""
        for node in self.get_memory():
            if node.id == node_id:
                return node
        return None

    def clear_memory(self):
        """Clears memory for the current user."""
        if self.user_id in self.noesis_triad.context_synthesizer.long_term_memory:
            self.noesis_triad.context_synthesizer.long_term_memory[self.user_id].clear()

    def save_memory_to_file(self, filepath: str):
        """Saves the current user's memory to a JSON file."""
        memory_to_save = {self.user_id: self.get_memory()}
        with open(filepath, 'w') as f:
            json.dump(memory_to_save, f, default=lambda o: o.model_dump(mode='json') if hasattr(o, 'model_dump') else str(o), indent=4)

    def load_memory_from_file(self, filepath: str):
        """Loads memory for the current user from a JSON file."""
        with open(filepath, 'r') as f:
            loaded_memory_raw = json.load(f)
        
        user_memory = loaded_memory_raw.get(self.user_id, [])
        self.noesis_triad.context_synthesizer.long_term_memory[self.user_id] = [MemoryNode.parse_obj(node_dict) for node_dict in user_memory]

    def get_config(self):
        """Retrieves the full system configuration."""
        from config_loader import config_loader
        return config_loader.get_full_config()