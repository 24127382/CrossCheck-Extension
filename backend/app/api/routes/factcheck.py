"""API routes for factcheck"""
from fastapi import APIRouter, HTTPException
from ...schemas.request import FactCheckRequestSchema
from ...schemas.response import FactCheckResponse
from ...pipelines.factcheck_pipeline import factcheck_pipeline

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
        print(f"\n[API] Received fact-check request for: {request.text[:100]}")
        
        # result = await factcheck_pipeline.process(request.text, request.context)
        result = await factcheck_pipeline.run_factcheck_pipeline(request.text)
        
        print(f"[API] ✅ Fact-check completed successfully")
        return result
        
    except HTTPException as http_err:
        print(f"[API] ❌ HTTP Exception: {http_err.detail}")
        raise
    except Exception as e:
        print(f"[API] ❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return more detailed error message for debugging
        error_msg = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)
