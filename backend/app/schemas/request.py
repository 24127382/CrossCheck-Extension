from pydantic import BaseModel, Field
from typing import Optional

class FactCheckRequestSchema(BaseModel):
    """Fact-check request schema"""
    text: str = Field(..., min_length=1, max_length=5000, description="Claim to fact-check")
    context: Optional[str] = Field(None, max_length=10000, description="Additional context")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The Earth is flat",
                "context": "Scientific discussion about planetary shape"
            }
        }
