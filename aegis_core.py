# aegis_core.py

from noesis_triad import noesis_triad
from praxis_triad import praxis_triad
from wgpmhi import wgpmhi
from data_structures import UserProfile
from cognitive_packet_generator import cognitive_packet_generator
from isp_protocol import isp
from prometheus_iop import prometheus_iop

class AEGIS_Core:
    """
    The core abstraction layer for the AEGIS engine. Encapsulates the entire
    cognitive pipeline into a single, manageable component.
    """
    def __init__(self):
        # Instantiate all core components of the AEGIS architecture.
        self.noesis_triad = noesis_triad
        self.praxis_triad = praxis_triad
        self.wgpmhi = wgpmhi
        self.cognitive_packet_generator = cognitive_packet_generator
        self.isp = isp
        self.prometheus_iop = prometheus_iop

    def process_prompt(self, user_id: str, prompt: str, user_profile: UserProfile) -> dict:
        """
        Processes a user prompt through the entire AEGIS lifecycle and returns
        a dictionary containing all generated artifacts.
        """
        # Store the provided user profile.
        self.noesis_triad.context_synthesizer.user_profiles[user_id] = user_profile

        # Generate a blueprint using the Noesis Triad
        blueprint = self.noesis_triad.generate_blueprint(user_id, prompt)

        # Generate an output using the Praxis Triad
        generation_result = self.praxis_triad.generate_output(blueprint, user_profile)
        output = generation_result["output"]
        debug_report = generation_result["debug_report"]

        # Run the WGPMHI tests to get the results needed for the Cognitive Packet.
        # Pass None for packet initially as it hasn't been generated yet.
        wgpmhi_results = self.wgpmhi.run_tests(user_profile, blueprint, output, self.noesis_triad, None)

        # Generate a Cognitive Packet for training.
        cognitive_packet = self.cognitive_packet_generator.generate_packet(blueprint, output, wgpmhi_results, debug_report)

        # Store the Cognitive Packet in Long-Term Memory.
        self.noesis_triad.context_synthesizer.update_long_term_memory(user_id, cognitive_packet)

        # --- Project PROMETHEUS: Final Output and External Audit ---
        # 1. Simulate external audit via ISP (existing Project PANDORA)
        external_request = {"packet_id": cognitive_packet.packet_id}
        isp_response = self.isp.handle_external_audit_request(external_request, self.noesis_triad)

        # 2. Simulate sending the Cognitive Packet to the Prometheus monitoring system (NEW)
        prometheus_queue_status = self.prometheus_iop.send_cognitive_packet(cognitive_packet)
        # Return all generated artifacts in a structured dictionary.
        return {
            "final_output": output,
            "blueprint": blueprint,
            "wgpmhi_results": wgpmhi_results,
            "orthrus_debug_report": debug_report,
            "cognitive_packet": cognitive_packet,
            "isp_audit_response": isp_response,
            "prometheus_queue_status": prometheus_queue_status,
        }