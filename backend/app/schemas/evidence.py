from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Evidence:
    """Evidence item structure"""
    source: str
    stance: str = "neutral"  # 'supports', 'contradicts', 'neutral'
    score: float = 0.0
    text: str = ""
    sentence_id: Optional[int] = None
# EvidenceScore có vấn đề trong ranking service line 36 và factcheck pipeline line 64, cần sửa lại để tránh lỗi khi gán giá trị final_score
@dataclass
class EvidenceScore:
    evidence: Evidence
    relevance_score: float
    entailment_score: float = 0.0
    final_score: float = 0.0
    
    
@dataclass
class FactCheckEvidenceResponse:
    """Structured response for retreived evidence then 
    give it to llm to summarize and give verdict (use Evidence
    """
    claim: str
    evidences: List[Evidence]
    
@dataclass
class Node:
    title: str
    text: str
    depth: int
    score: float
    parent: str = None
