"""API routes for factcheck"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ...schemas.request import FactCheckRequestSchema
from ...schemas.response import FactCheckResponse
from ...pipelines.factcheck_pipeline import factcheck_pipeline
from ...services.cache_service import cache_service
from ...services.llm_service import llm_service

router = APIRouter(prefix="/api", tags=["factcheck"])

class ExplainRequest(BaseModel):
    """Request body for /api/explain"""
    claim: str

class ExplainResponse(BaseModel):
    """Response for /api/explain"""
    claim: str
    verdict: str
    confidence: float
    explanation: str

@router.post("/factcheck", response_model=FactCheckResponse)
async def factcheck(request: FactCheckRequestSchema) -> FactCheckResponse:
    """
    Fact-check a claim using RoBERTa model
    
    Request:
        - text: The claim to fact-check
        - context: Optional context information
        
    Response:
        - verdict: SUPPORTS, CONTRADICTS, NOT_ENOUGH_INFO
        - confidence: Confidence score (0-1)
        - evidences: List of evidence items
        - summary: Empty string (use /api/explain for explanation)
    """
    try:
        print(f"\n[API] Received fact-check request for: {request.text[:100]}")
        
        # Use the full pipeline with RoBERTa verdict
        result = await factcheck_pipeline.process(request.text, request.context)
        
        print(f"[API] ✅ Fact-check completed successfully")
        return result
        
    except HTTPException as http_err:
        print(f"[API] ❌ HTTP Exception: {http_err.detail}")
        raise
    except Exception as e:
        print(f"[API] ❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest) -> ExplainResponse:
    """
    Generate a natural language explanation for a fact-checked claim
    
    Uses cached result from a previous /api/factcheck request.
    Does NOT perform retrieval, ranking, or RoBERTa again.
    
    Request:
        - claim: The claim to explain
        
    Response:
        - claim: The claim
        - verdict: The verdict from RoBERTa
        - confidence: Confidence from RoBERTa
        - explanation: Natural language explanation from Gemini
    """
    try:
        print(f"\n[API] Received explanation request for: {request.claim[:100]}")
        
        # Try to get from cache
        cached_result = cache_service.get(request.claim)
        if not cached_result:
            print(f"[API] ❌ No cached result for claim: {request.claim[:50]}")
            raise HTTPException(
                status_code=400,
                detail="No fact-check result found. Please run /api/factcheck first."
            )
        
        print(f"[API] ✅ Found cached result")
        
        # Generate explanation using Gemini with cached evidence
        print(f"[API] Calling Gemini for explanation...")
        gemini_result = await llm_service.summarize(request.claim, cached_result.evidences)
        explanation = gemini_result.get("summary", "")
        
        print(f"[API] ✅ Explanation generated")
        
        response = ExplainResponse(
            claim=request.claim,
            verdict=cached_result.verdict,
            confidence=cached_result.confidence,
            explanation=explanation
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] ❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)
