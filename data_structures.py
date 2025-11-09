# data_structures.py

from typing import List, Dict, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

class CognitivePacket(BaseModel):
    """Represents the complete data snapshot of a single cognitive cycle."""
    packet_id: str
    timestamp: datetime
    intent: Dict[str, Any]
    output_summary: str
    wgpmhi_results: Dict[str, Any]
    debug_report: str


class MathLanguageTag(BaseModel):
    type: str
    value: str

class Blueprint(BaseModel):
    """The strategic plan formulated by the Noesis Triad."""
    packet_id: str
    primary_intent: str
    latent_intent: str
    tags: List[MathLanguageTag]
    constraints: List[str]
    fallacies: List[str]
    expected_outcome: str
    ethical_considerations: str
    risk_score: float
    confidence_score: float
    novelty_score: float
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
        populate_by_name = True

class TimeDataSeries(BaseModel):
    """
    A structured container for sequential, time-series data, used for
    predictive modeling and trend analysis by Project CHRONOS.
    """
    series_id: str
    data_points: List[Tuple[datetime, float]]
    metadata: Dict[str, Any] = {}