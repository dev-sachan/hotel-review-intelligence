"""Output contracts (Pydantic).

Two shapes:
  * `SubmissionOutput`  — byte-for-byte the structure of the provided sample_output.json,
    so our results validate against the hackathon's expected schema.
  * `RichRecommendation` / `RichOutput` — the same ranking plus the evidence, matched
    aspects, justification and caveats that make the result trustworthy and demo-able.

Every output the pipeline writes is validated through these models, so a malformed
recommendation can never reach a file or the app.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ----------------------------------------------------------- submission (sample schema)
class TopHotel(BaseModel):
    rank: int = Field(ge=1, le=5)
    hotel_id: str
    hotel_name: str
    hotel_category: str
    score: float


class SubmissionOutput(BaseModel):
    profile_id: str
    archetype: str
    desired_dims: list[str]
    top_hotels: list[TopHotel] = Field(min_length=1, max_length=5)


# ----------------------------------------------------------- rich (evidence-carrying)
class Evidence(BaseModel):
    review_id: str
    quote: str
    aspect: str
    date: str
    verified: bool
    sentiment: float


class RichRecommendation(BaseModel):
    rank: int = Field(ge=1, le=5)
    hotel_id: str
    hotel_name: str
    hotel_category: str
    score: float
    matched_aspects: list[str]
    aspect_scores: dict[str, float]
    justification: str = Field(max_length=800)
    supporting_evidence: list[Evidence] = Field(min_length=1, max_length=3)
    caveats: Optional[str] = None


class RichOutput(BaseModel):
    profile_id: str
    archetype: str
    profile_summary: str
    desired_dims: list[str]
    recommendations: list[RichRecommendation] = Field(min_length=1, max_length=5)
