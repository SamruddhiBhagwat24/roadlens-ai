"""Structured road state representation and tracking."""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RoadState(BaseModel):
    """Cumulative state of perceived road environment."""
    current_speed_limit_mph: Optional[int] = None
    advisory_speed_mph: Optional[int] = None
    stop_sign_detected: bool = False
    warning_active: bool = False
    warning_description: Optional[str] = None
    license_plates: List[str] = Field(default_factory=list)
    adverse_conditions: List[str] = Field(default_factory=list)
    overall_confidence: float = 1.0
