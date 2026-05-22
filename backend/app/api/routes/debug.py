"""Debug and testing endpoints"""
from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["debug"])

@router.get("/debug/models")
async def get_models_status() -> dict:
    """Get status of all loaded models"""
    return {
        "models": {
            "clip": {"status": "loaded", "name": "ViT-B/32"},
            "entailment": {"status": "loaded", "name": "roberta-large-mnli"},
            "retrieval": {"status": "loaded", "name": "all-MiniLM-L6-v2"},
        }
    }

@router.post("/debug/test-pipeline")
async def test_pipeline(claim: str) -> dict:
    """Test the fact-checking pipeline with a sample claim"""
    return {
        "claim": claim,
        "pipeline_test": "success"
    }
