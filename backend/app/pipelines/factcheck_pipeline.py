"""Factcheck pipeline - orchestrates the fact-checking flow"""
from typing import Optional
from ..schemas.response import FactCheckResponse, EvidenceSchema
from ..services.retrieval_service import retrieval_service
from ..services.entailment_service import entailment_service
from ..services.ranking_service import ranking_service
from ..services.llm_service import llm_service
from ..services.clip_service import clip_service
from ..schemas.evidence import Evidence
class FactCheckPipeline:
    """Main fact-checking pipeline"""
    
    async def process(self, claim: str, context: Optional[str] = None) -> FactCheckResponse:
        """
        Main pipeline: claim → retrieve → rank → entailment → filter → summarize → response
        
        Flow:
        1. Retrieve relevant evidence documents from Wikipedia
        2. Rank evidence by relevance using embeddings
        3. Compute entailment scores for ranked evidence
        4. Filter by combined threshold
        5. Generate verdict and summary using LLM
        6. Format and return response
        
        Args:
            claim: The claim to fact-check
            context: Optional context information
            
        Returns:
            FactCheckResponse with verdict, confidence, summary, and evidences
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
                    summary="Could not find evidence on Wikipedia for this claim.",
                    evidences=[]
                )
            
            # Step 2: Rank evidence by relevance
            print("[Pipeline] Step 2: Ranking evidence by relevance...")
            ranked_evidences = await ranking_service.rank_evidence(claim, evidences)
            print(f"[Pipeline] ✅ Ranked evidence")
            
            # Step 3: Compute entailment scores
            print("[Pipeline] Step 3: Computing entailment scores...")
            entailment_scored = await entailment_service.compute_entailment(claim, evidences)
            print(f"[Pipeline] ✅ Computed entailment scores")
            
            # Merge entailment scores into ranked evidence
            # Create a mapping for easy lookup
            entailment_map = {e.evidence.source: e.entailment_score for e in entailment_scored}
            for scored_evidence in ranked_evidences:
                source = scored_evidence.evidence.source
                if source in entailment_map:
                    # Combine relevance + entailment scores
                    scored_evidence.entailment_score = entailment_map[source]
                    scored_evidence.final_score = (scored_evidence.relevance_score + entailment_map[source]) / 2
            
            # Step 4: Filter by threshold
            print("[Pipeline] Step 4: Filtering evidence by threshold...")
            filtered_evidences = await ranking_service.filter_evidence(ranked_evidences, threshold=0.3)
            print(f"[Pipeline] ✅ Filtered: {len(ranked_evidences)} → {len(filtered_evidences)}")
            
            if not filtered_evidences:
                print("[Pipeline] ⚠️ No evidence passed threshold!")
                return FactCheckResponse(
                    claim=claim,
                    verdict="NOT_ENOUGH_INFO",
                    confidence=0.0,
                    summary="Found evidence but scores too low to use.",
                    evidences=[]
                )
            
            # Step 5: Summarize using Gemini with top evidence
            print("[Pipeline] Step 5: Calling Gemini API for analysis...")
            top_evidences = [e.evidence for e in filtered_evidences[:3]]  # Use top 3 evidences
            llm_result = await llm_service.summarize(claim, top_evidences)
            print(f"[Pipeline] ✅ Gemini response: {llm_result}")
            
            # Step 6: Format response
            print("[Pipeline] Step 6: Formatting response...")
            response = FactCheckResponse(
                claim=claim,
                verdict=llm_result.get("verdict", "NOT_ENOUGH_INFO"),
                confidence=llm_result.get("confidence", 0.0),
                summary=llm_result.get("summary", ""),
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
            print("[Pipeline] ✅ Pipeline complete!")
            return response
            
        except Exception as e:
            print(f"[Pipeline] ❌ FATAL ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    
factcheck_pipeline = FactCheckPipeline()
