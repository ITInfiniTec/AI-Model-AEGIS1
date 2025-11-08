# api_models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any

class UserProfileValues(BaseModel):
    """Defines the schema for user profile values."""
    controversial_topics_approach: float = Field(..., ge=0.0, le=1.0)
    importance_of_accuracy: float = Field(..., ge=0.0, le=1.0)

class UserProfileData(BaseModel):
    """Defines the schema for user profile data in the request."""
    values: UserProfileValues
    passions: List[str] = []

class ProcessRequest(BaseModel):
    """Defines the schema for the main /process API request."""
    user_id: str
    prompt: str
    user_profile_data: UserProfileData