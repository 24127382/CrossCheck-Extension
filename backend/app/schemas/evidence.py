from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Evidence:
    """Evidence item structure"""
    source: str
    stance: str = "neutral"  # 'supports', 'contradicts', 'neutral'
    score: float = 0.0
    text: str = ""

@dataclass
class EvidenceScore:
    evidence: Evidence
    relevance_score: float
    sentence_score: float = 0.0
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
class EvidenceNode:
    """Node structure for evidence graph"""
    id: str
    title: str
    text: str
    
    parent: Optional[str] = None  # ID of parent node
    children: List[str] = None  # IDs of child nodes
    
    depth: int = 0  # Depth in the graph
    relevance_score: float = 0.0