# cognitive_packet_generator.py

import uuid
from typing import Dict, Any
from datetime import datetime
from data_structures import CognitivePacket, Blueprint

class CognitivePacketGenerator:
    def __init__(self):
        pass

    def generate_packet(self, blueprint: Blueprint, final_output: str, wgpmhi_results: Dict[str, Any], debug_report: Dict[str, Any]) -> CognitivePacket:
        """
        Assembles a CognitivePacket from the results of a single interaction,
        simulating the data generation process for training Prometheus agents.
        """
        # Extract a summary of the final output, excluding the detailed execution plan.
        output_summary = final_output.split("--- SIMULATED PROSE OUTPUT ---")[-1].strip()

        packet = CognitivePacket(
            packet_id=f"cp-{uuid.uuid4()}",
            timestamp=datetime.now(),
            intent={
                "primary": blueprint.primary_intent,
                "latent": blueprint.latent_intent,
            },
            output_summary=output_summary,
            wgpmhi_results=wgpmhi_results,
            # The debug report is a dictionary; we'll serialize it to a string for the packet.
            debug_report=str(debug_report),
        )
        return packet

cognitive_packet_generator = CognitivePacketGenerator()