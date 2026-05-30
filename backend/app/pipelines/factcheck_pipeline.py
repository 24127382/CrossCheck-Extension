"""Factcheck pipeline - orchestrates the fact-checking flow"""
from typing import Optional
from ..schemas.response import FactCheckResponse, EvidenceSchema
from ..services.retrieval_service import retrieval_service
from ..services.entailment_service import entailment_service
from ..services.ranking_service import ranking_service
from ..services.llm_service import llm_service
from ..services.clip_service import clip_service
# 2 file dưới for test only
from .llm_service import summarize
from .retrieve_service import retrieve
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

    async def run_factcheck_pipeline(self, claim: str) -> FactCheckResponse:
        """Real pipeline - retrieves from Wikipedia and summarizes with Gemini"""
        try:
            print(f"\n[Pipeline] Starting fact-check for: {claim[:100]}")
            
            # Step 1: Retrieve evidence from Wikipedia
            print("[Pipeline] Step 1: Retrieving evidence from Wikipedia...")
            try:
                retrieved_evidences = await retrieve(claim)
                print(f"[Pipeline] Retrieved {len(retrieved_evidences)} evidence items")
                if not retrieved_evidences:
                    print("[Pipeline] ⚠️ No evidence retrieved!")
                    return FactCheckResponse(
                        claim=claim,
                        verdict="NOT_ENOUGH_INFO",
                        confidence=0.0,
                        summary="Could not find evidence on Wikipedia for this claim.",
                        evidences=[]
                    )
                
                # Convert FactCheckEvidenceResponse to Evidence dataclass for summarize
                from ..schemas.evidence import Evidence
                evidences = [
                    Evidence(
                        source=e.source,
                        stance="NEUTRAL",
                        score=0.0,
                        text=e.content
                    )
                    for e in retrieved_evidences
                ]
            except Exception as e:
                print(f"[Pipeline] ❌ Retrieval error: {str(e)}")
                import traceback
                traceback.print_exc()
                raise Exception(f"Failed to retrieve evidence: {str(e)}")
            
            # Step 2: Summarize using Gemini
            print("[Pipeline] Step 2: Calling Gemini API for analysis...")
            try:
                llm_result = await summarize(claim, evidences)
                print(f"[Pipeline] ✅ Gemini response: {llm_result}")
            except Exception as e:
                print(f"[Pipeline] ❌ Gemini error: {str(e)}")
                import traceback
                traceback.print_exc()
                raise Exception(f"Failed to analyze with Gemini: {str(e)}")
            
            # Step 3: Format response
            print("[Pipeline] Step 3: Formatting response...")
            response = FactCheckResponse(
                claim=claim,
                verdict=llm_result.get("verdict", "NOT_ENOUGH_INFO"),
                confidence=llm_result.get("confidence", 0.0),
                summary=llm_result.get("summary", ""),
                evidences=[
                    EvidenceSchema(
                        source=e.source,
                        stance="NEUTRAL",
                        score=0.0,
                        text=e.text
                    )
                    for e in evidences
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
