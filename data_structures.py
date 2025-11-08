```python
# data_structures.py

from typing import List, Dict, Any, Tuple
from datetime import datetime

# Define the structure for Math Language Tags
MathLanguageTag = Dict[str, str]

# Define the structure for the Cognitive Packet
class CognitivePacket:
    def __init__(
        self,
        packet_id: str,
        scenario: str,
        intent: Dict[str, Any],
        ethical_considerations: Dict[str, List[str]],
        ideal_response: Dict[str, Any],
        wgpmhi_results: Dict[str, Any],
        debug_report: Dict[str, Any],
    ):
        self.packet_id = packet_id
        self.scenario = scenario
        self.intent = intent
        self.ethical_considerations = ethical_considerations
        self.ideal_response = ideal_response
        self.wgpmhi_results = wgpmhi_results
        self.debug_report = debug_report

# Define the structure for the Blueprint
class Blueprint:
    def __init__(
        self,
        primary_intent: str,
        latent_intent: str,
        tags: List[MathLanguageTag],
        constraints: List[str],
        expected_outcome: str,
        ethical_considerations: str,
        ambiguity_analysis: Dict[str, List[str]],
        persona: str,
        user_id: str,
    ):
        self.primary_intent = primary_intent
        self.latent_intent = latent_intent
        self.tags = tags
        self.constraints = constraints
        self.expected_outcome = expected_outcome
        self.ethical_considerations = ethical_considerations
        self.ambiguity_analysis = ambiguity_analysis
        self.persona = persona
        self.user_id = user_id


# Define the structure for UserProfile
class UserProfile:
    def __init__(self, user_id: str, values: Dict[str, Any]):
        self.user_id = user_id
        self.values = values
        self.passions: List[str] = [] # Field for persona-driven simulation

# Define the structure for the advanced memory system (Project MNEMOSYNE)
class MemoryNode:
    def __init__(
        self,
        node_id: str,
        timestamp: datetime,
        core_intent_vector: List[float],
        keywords: List[str],
        performance_score: float,
        packet_reference: CognitivePacket,
    ):
        self.id = node_id
        self.timestamp = timestamp
        self.core_intent_vector = core_intent_vector
        self.keywords = keywords
        self.performance_score = performance_score
        self.packet_reference = packet_reference

# Define the structure for time-series data (Project CHRONOS)
class TimeDataSeries:
    """
    A structured container for sequential, time-series data, used for
    predictive modeling and trend analysis by Project CHRONOS.
    """
    def __init__(self, series_id: str, data_points: List[Tuple[datetime, float]], metadata: Dict[str, Any] = None):
        self.series_id = series_id
        # Data points are stored as (timestamp, value) tuples.
        self.data_points = data_points 
        self.metadata = metadata if metadata is not None else {}


```