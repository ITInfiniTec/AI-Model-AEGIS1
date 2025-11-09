# data_structures.py

from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class CognitivePacket(BaseModel):
    """Represents the complete data snapshot of a single cognitive cycle."""
    packet_id: str
    timestamp: datetime
    intent: Dict[str, Any]
    risk_score: float
    novelty_score: float
    output_summary: str
    wgpmhi_results: Dict[str, Any]
    debug_report: str


class SemanticTag(BaseModel):
    """Represents a semantic tag extracted from a prompt or context."""
    type: str
    value: str

class Blueprint(BaseModel):
    """The strategic plan formulated by the Noesis Triad."""
    packet_id: str
    original_intent: str
    primary_intent: str
    latent_intent: str
    tags: List[SemanticTag]
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
    external_data: Optional[str] = None

class UserProfile(BaseModel):
    """Defines the user's characteristics and preferences."""
    user_id: str
    values: Dict[str, Any]
    passions: List[str] = []

class MemoryNode(BaseModel):
    """A node in the long-term memory graph, representing a past interaction."""
    node_id: str
    timestamp: datetime
    core_intent_vector: List[float]
    keywords: List[str]
    performance_score: float
    packet_reference: CognitivePacket

class ExecutionPlan(BaseModel):
    """
    A structured representation of the compiled plan from the UniversalCompiler.
    This replaces the brittle QVC string format with a robust data contract.
    """
    intent_hash: int
    constraints: List[str]
    target_format: str
    target_audience: str
    fallacy_warnings: List[str]
    external_data_required: bool
    word_limit: Optional[int] = None
    safety_priority: str
    ethical_consult_required: bool
    simulated_forecast_result: str
    stg: Dict[str, Dict[str, Any]]

class TimeDataSeries(BaseModel):
    """
    A structured container for sequential, time-series data, used for
    predictive modeling and trend analysis by Project CHRONOS.
    """
    series_id: str
    data_points: List[Tuple[datetime, float]]
    metadata: Dict[str, Any] = {}