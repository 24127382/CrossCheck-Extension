"""Ranking service - handles evidence ranking by using all-MiniLM-L6-v2 embeddings and other factors"""
from typing import List
from sentence_transformers import SentenceTransformer, util
from ..schemas.evidence import Evidence, EvidenceScore

class RankingService:
    """Service for ranking and filtering evidence"""
    
    def __init__(self):
        # Initialize embedding model for relevance scoring
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def rank_evidence(self, claim: str, evidences: List[Evidence]) -> List[EvidenceScore]:
        """
        Rank evidence by relevance using embeddings (all-MiniLM-L6-v2)
        
        Args:
            claim: The claim to evaluate
            evidences: List of candidate evidence items
            
        Returns:
            Ranked list of EvidenceScore objects sorted by relevance
        """
        print(f"[Ranking] Computing embeddings for claim and {len(evidences)} evidences...")
        
        # Encode claim
        claim_embedding = self.model.encode(claim, convert_to_tensor=True)
        
        # Encode all evidence texts
        evidences_embeddings = [self.model.encode(e.text, convert_to_tensor=True) for e in evidences]

        # Compute cosine similarity scores
        cosine_scores = [util.cos_sim(claim_embedding, e_emb)[0][0].item() for e_emb in evidences_embeddings]
        
        print(f"[Ranking] Cosine scores: {[f'{s:.3f}' for s in cosine_scores]}")
        
        results = []
        for i, relevance_score in enumerate(cosine_scores):
            evidence_score = EvidenceScore(
                evidence=evidences[i],
                relevance_score=relevance_score,
                entailment_score=0.0,  # Placeholder - can be computed by entailment_service later
                final_score=relevance_score  # For now, use relevance_score as final_score
            )
            results.append(evidence_score)
        
        # Sort by final score (highest first)
        sorted_results = sorted(results, key=lambda x: x.final_score, reverse=True)
        print(f"[Ranking] Top 3 scores after ranking: {[f'{r.final_score:.3f}' for r in sorted_results[:3]]}")
        return sorted_results
    
    async def filter_evidence(self, scored_evidences: List[EvidenceScore], 
                            threshold: float = 0.3) -> List[EvidenceScore]:
        """
        Filter evidence by confidence threshold
        
        Args:
            scored_evidences: List of ranked evidence scores
            threshold: Minimum score to keep evidence (0.0 to 1.0)
            
        Returns:
            Filtered list of EvidenceScore objects above threshold
        """
        filtered = [e for e in scored_evidences if e.final_score >= threshold]
        print(f"[Ranking] Filtered: {len(scored_evidences)} → {len(filtered)} (threshold: {threshold})")
        return filtered

ranking_service = RankingService()
