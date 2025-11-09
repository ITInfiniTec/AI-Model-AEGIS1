# data_structures.py

from typing import List, Dict, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

class CognitivePacket(BaseModel):
    """Represents the complete data snapshot of a single cognitive cycle."""
    packet_id: str
    scenario: str
    intent: Dict[str, Any]
    ethical_considerations: Dict[str, List[str]]
    ideal_response: Dict[str, Any]
    wgpmhi_results: Dict[str, Any]
    debug_report: Dict[str, Any]


class MathLanguageTag(BaseModel):
    type: str
    value: str

class Blueprint(BaseModel):
    """The strategic plan formulated by the Noesis Triad."""
    primary_intent: str
    latent_intent: str
    tags: List[MathLanguageTag]
    constraints: List[str]
    expected_outcome: str
    ethical_considerations: str
    ambiguity_analysis: Dict[str, List[str]]
    persona: str
    user_id: str

class UserProfile(BaseModel):
    """Defines the user's characteristics and preferences."""
    user_id: str
    values: Dict[str, Any]
    passions: List[str] = []

class MemoryNode(BaseModel):
    """A node in the long-term memory graph, representing a past interaction."""
    id: str = Field(..., alias="node_id")
    timestamp: datetime
    core_intent_vector: List[float]
    keywords: List[str]
    performance_score: float
    packet_reference: CognitivePacket

    class Config:
        allow_population_by_field_name = True

class TimeDataSeries(BaseModel):
    """
    A structured container for sequential, time-series data, used for
    predictive modeling and trend analysis by Project CHRONOS.
    """
    series_id: str
    data_points: List[Tuple[datetime, float]]
    metadata: Dict[str, Any] = {}