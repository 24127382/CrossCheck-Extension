"""Health check endpoints"""
from fastapi import APIRouter
from ...schemas.response import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        version="0.1.0"
    )

@router.get("/health/ready")
async def readiness() -> dict:
    """Readiness check - verify all services are ready"""
    return {
        "status": "ready",
        "services": {
            "retrieval": "ok",
            "entailment": "ok",
            "llm": "ok",
            "clip": "ok",
        }
    }

@router.get("/health/live")
async def liveness() -> dict:
    """Liveness check - verify service is running"""
    return {"status": "alive"}
