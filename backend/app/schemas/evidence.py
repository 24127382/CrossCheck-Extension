from dataclasses import dataclass
from typing import List

@dataclass
class Evidence:
    """Evidence item structure"""
    source: str
    stance: str  # 'supports', 'contradicts', 'neutral'
    score: float
    text: str = ""

@dataclass
class EvidenceScore:
    """Scored evidence for ranking"""
    evidence: Evidence
    relevance_score: float
    entailment_score: float
    final_score: float

@dataclass
class FactCheckEvidenceResponse:
    """Structured response for retreived evidence then 
    give it to llm to summarize and give verdict (use Evidence
    """
    claim: str
    evidences: List[Evidence]