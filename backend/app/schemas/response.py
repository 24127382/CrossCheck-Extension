from pydantic import BaseModel, Field
from typing import List, Optional

class EvidenceSchema(BaseModel):
    """Evidence item for fact-checking result"""
    source: str = Field(..., description="Source of the evidence")
    stance: str = Field(..., description="Stance towards claim: supports, contradicts, neutral")
    score: float = Field(..., ge=0, le=1, description="Confidence score for this evidence")
    text: Optional[str] = Field(None, description="Full text of the evidence")

class FactCheckRequest(BaseModel):
    """Request body for fact-check endpoint"""
    text: str = Field(..., description="Claim to fact-check")
    context: Optional[str] = Field(None, description="Additional context about the claim")

class FactCheckResponse(BaseModel):
    """Response from fact-check endpoint"""
    claim: str = Field(..., description="Original claim")
    verdict: str = Field(..., description="Verdict: REFUTED, SUPPORTED, NOT_ENOUGH_INFO, DISPUTED")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the verdict")
    summary: str = Field(..., description="Summary of the fact-check result")
    evidences: List[EvidenceSchema] = Field(..., description="List of evidence items")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
