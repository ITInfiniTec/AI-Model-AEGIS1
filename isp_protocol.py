# isp_protocol.py

from typing import Dict, Any
from noesis_triad import NoesisTriad

class InterSystemProtocol:
    """Simulates an endpoint for handling secure, asynchronous inter-system communication."""

    def handle_external_audit_request(self, request: Dict[str, Any], noesis_triad: NoesisTriad) -> Dict[str, Any]:
        """
        Simulates receiving an external request for an audit report on a past interaction.
        """
        packet_id = request.get("packet_id")
        if not packet_id:
            return {"status": "error", "message": "Request is missing 'packet_id'."}

        # Search through all users' long-term memory for the requested packet.
        # In a real system, this would be a direct database lookup.
        for user_id, memory_nodes in noesis_triad.context_synthesizer.long_term_memory.items():
            for node in memory_nodes:
                if node.packet_reference.packet_id == packet_id:
                    # Found the packet, return its debug report.
                    return {"status": "success", "packet_id": packet_id, "report": node.packet_reference.debug_report}

        return {"status": "error", "message": f"Packet with ID '{packet_id}' not found in long-term memory."}

# Singleton instance
isp = InterSystemProtocol()