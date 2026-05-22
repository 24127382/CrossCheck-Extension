"""API routes for factcheck"""
from fastapi import APIRouter, HTTPException
from app.schemas.request import FactCheckRequestSchema
from app.schemas.response import FactCheckResponse
from app.pipelines.factcheck_pipeline import factcheck_pipeline

router = APIRouter(prefix="/api", tags=["factcheck"])

@router.post("/factcheck", response_model=FactCheckResponse)
async def factcheck(request: FactCheckRequestSchema) -> FactCheckResponse:
    """
    Fact-check a claim
    
    Request:
        - text: The claim to fact-check
        - context: Optional context information
        
    Response:
        - verdict: REFUTED, SUPPORTED, NOT_ENOUGH_INFO, DISPUTED
        - confidence: Confidence score (0-1)
        - summary: Summary of the fact-check
        - evidences: List of evidence items
    """
    try:
        result = await factcheck_pipeline.process(request.text, request.context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
