# cognitive_packet_generator.py

import uuid
from typing import Dict, Any
from data_structures import CognitivePacket, Blueprint

class CognitivePacketGenerator:
    def __init__(self):
        pass

    def generate_packet(self, blueprint: Blueprint, final_output: str, wgpmhi_results: Dict[str, Any], debug_report: Dict[str, Any]) -> CognitivePacket:
        """
        Assembles a CognitivePacket from the results of a single interaction,
        simulating the data generation process for training Prometheus agents.
        """
        packet_id = f"cp-{uuid.uuid4()}"
        scenario = f"User prompt was: '{blueprint.primary_intent}'. The system's latent intent was to: '{blueprint.latent_intent}'."

        # This simulates the ideal reasoning. In a real training scenario, this might be human-verified or refined.
        reasoning = f"The system identified the primary intent, inferred the latent intent, and generated a plan. WGPMHI audit results: {wgpmhi_results.get('self_correction_audit', 'N/A')}"

        packet = CognitivePacket(
            packet_id=packet_id,
            scenario=scenario,
            intent={
                "raw_prompt": blueprint.primary_intent,
                "inferred_goal": blueprint.latent_intent,
                "math_language_tags": [tag['value'] for tag in blueprint.tags],
            },
            ethical_considerations={
                "potential_dilemmas": [blueprint.ethical_considerations],
                "cmep_alignment": "Response was generated under CMEP guidelines, checking for red-lines and user-value alignment."
            },
            ideal_response={
                "persona": "The_Architect", # This could be dynamic in a more advanced implementation
                "content": final_output,
                "reasoning": reasoning,
            },
            wgpmhi_results=wgpmhi_results,
            debug_report=debug_report,
        )
        return packet

cognitive_packet_generator = CognitivePacketGenerator()