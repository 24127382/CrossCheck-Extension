"""Factcheck pipeline - orchestrates the fact-checking flow"""
from typing import Optional
from app.schemas.response import FactCheckResponse, EvidenceSchema
from app.services.retrieval_service import retrieval_service
from app.services.entailment_service import entailment_service
from app.services.ranking_service import ranking_service
from app.services.llm_service import llm_service
from app.services.clip_service import clip_service

class FactCheckPipeline:
    """Main fact-checking pipeline"""
    
    async def process(self, claim: str, context: Optional[str] = None) -> FactCheckResponse:
        """
        Main pipeline: claim → retrieve → encode → rerank → summarize → response
        
        Flow:
        1. Retrieve relevant evidence documents
        2. Encode claim and evidence using embeddings/CLIP
        3. Compute entailment scores
        4. Rank evidence by multiple factors
        5. Generate verdict and summary
        6. Format response
        
        Args:
            claim: The claim to fact-check
            context: Optional context information
            
        Returns:
            FactCheckResponse with verdict, confidence, summary, and evidences
        """
        
        # Step 1: Retrieve evidence
        evidences = await retrieval_service.retrieve_evidence(claim, top_k=10)
        
        # Step 2: Compute entailment scores
        entailment_scores = []
        for evidence in evidences:
            scores = await entailment_service.compute_entailment(
                evidence.text, claim
            )
            entailment_scores.append(scores)
        
        # Step 3: Rank evidence
        # Combine similarity, entailment, and other factors
        ranked_evidences = await ranking_service.rank_evidence(claim, evidences)
        
        # Step 4: Filter by threshold
        filtered_evidences = await ranking_service.filter_evidence(ranked_evidences)
        
        # Step 5: Generate verdict and summary
        verdict_result = await llm_service.generate_verdict(
            claim, 
            " ".join([e.evidence.text for e in filtered_evidences])
        )
        
        summary = await llm_service.summarize(claim, filtered_evidences)
        
        # Step 6: Format response
        response = FactCheckResponse(
            claim=claim,
            verdict=verdict_result.get("verdict", "NOT_ENOUGH_INFO"),
            confidence=verdict_result.get("confidence", 0.0),
            summary=summary,
            evidences=[
                EvidenceSchema(
                    source=e.evidence.source,
                    stance=e.evidence.stance,
                    score=e.final_score,
                    text=e.evidence.text
                )
                for e in filtered_evidences[:5]
            ]
        )
        
        return response

factcheck_pipeline = FactCheckPipeline()
