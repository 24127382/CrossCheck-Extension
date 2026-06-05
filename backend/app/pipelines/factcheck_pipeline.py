"""Factcheck pipeline - orchestrates the fact-checking flow"""
from typing import Optional
from ..schemas.response import FactCheckResponse, EvidenceSchema
from ..services.retrieval_service import retrieval_service
from ..services.entailment_service import entailment_service
from ..services.ranking_service import ranking_service
from ..services.cache_service import cache_service
from ..schemas.evidence import Evidence
class FactCheckPipeline:
    """Main fact-checking pipeline"""
    
    async def process(self, claim: str, context: Optional[str] = None) -> FactCheckResponse:
        """
        Main fact-checking pipeline: claim → retrieve → rank → filter → RoBERTa verdict → response
        
        Flow:
        1. Retrieve relevant evidence documents from Wikipedia
        2. Rank evidence by relevance using embeddings
        3. Filter by confidence threshold
        4. Use RoBERTa model to predict verdict
        5. Cache result for later explanation
        6. Return response with verdict and confidence (no summary)
        
        Args:
            claim: The claim to fact-check
            context: Optional context information (unused in this version)
            
        Returns:
            FactCheckResponse with verdict, confidence, and top evidences
        """
        try:
            print(f"\n[Pipeline] Starting fact-check for: {claim[:100]}")
            
            # Step 1: Retrieve evidence from Wikipedia
            print("[Pipeline] Step 1: Retrieving evidence from Wikipedia...")
            evidences = await retrieval_service.retrieve_evidence(claim, top_k=10)
            print(f"[Pipeline] ✅ Retrieved {len(evidences)} evidence items")
            
            if not evidences:
                print("[Pipeline] ⚠️ No evidence retrieved!")
                return FactCheckResponse(
                    claim=claim,
                    verdict="NOT_ENOUGH_INFO",
                    confidence=0.0,
                    summary="",
                    evidences=[]
                )
            
            # Step 2: Rank evidence by relevance
            print("[Pipeline] Step 2: Ranking evidence by relevance...")
            ranked_evidences = await ranking_service.rank_evidence(claim, evidences)
            print(f"[Pipeline] ✅ Ranked evidence")
            
            # Step 3: Filter by threshold
            print("[Pipeline] Step 3: Filtering evidence by threshold...")
            filtered_evidences = await ranking_service.filter_evidence(ranked_evidences, threshold=0.3)
            print(f"[Pipeline] ✅ Filtered: {len(ranked_evidences)} → {len(filtered_evidences)}")
            
            if not filtered_evidences:
                print("[Pipeline] ⚠️ No evidence passed threshold!")
                return FactCheckResponse(
                    claim=claim,
                    verdict="NOT_ENOUGH_INFO",
                    confidence=0.0,
                    summary="",
                    evidences=[]
                )
            
            # Step 4: Use RoBERTa to predict verdict
            print("[Pipeline] Step 4: Running RoBERTa prediction...")
            top_evidences = [e.evidence for e in filtered_evidences[:5]]  # Use top 5 evidences
            verdict_result = await entailment_service.predict_verdict(claim, top_evidences)
            print(f"[Pipeline] ✅ RoBERTa verdict: {verdict_result['verdict']} (confidence: {verdict_result['confidence']:.3f})")
            
            # Step 5: Format response (no summary)
            print("[Pipeline] Step 5: Formatting response...")
            response = FactCheckResponse(
                claim=claim,
                verdict=verdict_result.get("verdict", "NOT_ENOUGH_INFO"),
                confidence=verdict_result.get("confidence", 0.0),
                summary="",  # No summary - RoBERTa only provides verdict
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
            
            # Step 6: Cache result for later explanation
            print("[Pipeline] Step 6: Caching result...")
            cache_service.save(
                claim=claim,
                verdict=response.verdict,
                confidence=response.confidence,
                evidences=top_evidences
            )
            
            print("[Pipeline] ✅ Pipeline complete!")
            return response
            
        except Exception as e:
            print(f"[Pipeline] ❌ FATAL ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    
factcheck_pipeline = FactCheckPipeline()
