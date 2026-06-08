"""Factcheck pipeline - orchestrates the fact-checking flow"""
from typing import Optional
from ..schemas.response import FactCheckResponse, EvidenceSchema
from ..services.retrieval_service import retrieval_service, query_builder, wiki_client
from ..services.entailment_service import entailment_service
from ..services.ranking_service import ranking_service, sentence_selector
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
            
            # Step 1: noun chunking 
            print("[Pipeline] Step 1: Extracting topics...")
            topics = query_builder.extract_topics(claim)
            # Step 2: Retrieve evidence from Wikipedia
            print(f"[Pipeline] Step 2: Retrieving evidence for topics: {topics}")
            evidences = retrieval_service.retrieve(topics)
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
            

            print("[Pipeline] Step 2: Ranking evidence by relevance...")
            ranked_evidences = await ranking_service.rank_evidence(claim, evidences)
            
            # Step 3: Tự lọc tại chỗ (thay cho hàm filter_evidence cũ)
            filtered_evidences = [e for e in ranked_evidences if e.final_score >= 0.3]
            if not filtered_evidences and ranked_evidences:
                filtered_evidences = ranked_evidences[:2]
                
            if not filtered_evidences and ranked_evidences:
                print("[Pipeline] ⚠️ Threshold quá cao! Lấy tạm Top 2 kết quả tốt nhất...")
                filtered_evidences = ranked_evidences[:2]
            
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
            # Step 4: filter evidence by top-k (e.g., top 5 documents)
            print("[Pipeline] Step 4: Selecting top-k documents...")
            top_evidences = [e.evidence for e in filtered_evidences[:5]]
            # Step 5: Tách câu và lấy Top N câu liên quan nhất từ tất cả các Document
            print("[Pipeline] Step 5: Extracting and ranking sentences...")
            # LƯU Ý: Truyền thẳng mảng top_evidences vào, không dùng list comprehension!
            clean_sentence_evidences = sentence_selector.select(
                evidences=top_evidences,
                top_n=1
            )

            # Nếu không trích xuất được câu nào, fallback trả lỗi
            if not clean_sentence_evidences:
                return FactCheckResponse(
                    claim=claim,
                    verdict="NOT_ENOUGH_INFO",
                    confidence=0.0,
                    summary="",
                    evidences=[]
                )

            best_sentence = clean_sentence_evidences[0].text
            print("[Pipeline] Step 6: Running RoBERTa prediction on pure sentences...")
            verdict_result = await entailment_service.predict_verdict(
                claim=claim,
                pseudo_outline=best_sentence
            )
            print(f"[Pipeline] ✅ RoBERTa verdict: {verdict_result['verdict']} (confidence: {verdict_result['confidence']:.3f})")
            
            
            # Step 7: Format response - Trả về những câu ngắn đã dùng để user dễ đọc
            print("[Pipeline] Step 7: Formatting response...")
            response = FactCheckResponse(
                claim=claim,
                verdict=verdict_result.get("verdict", "NOT_ENOUGH_INFO"),
                confidence=verdict_result.get("confidence", 0.0),
                summary=best_sentence,
                evidences=[
                    EvidenceSchema(
                        source=e.source,
                        stance=e.stance,
                        score=e.score,
                        text=e.text
                    )
                    for e in clean_sentence_evidences
                ]
            )
            
            # Step 8: Cache result for later explanation
            print("[Pipeline] Step 8: Caching result...")
            # cache_service.save(
            #     claim=claim,
            #     verdict=response.verdict,
            #     confidence=response.confidence,
            #     evidences=clean_sentence_evidences # Cache luôn câu ngắn cho gọn DB
            # )
            
            print("[Pipeline] ✅ Pipeline complete!")
            return response
            
        except Exception as e:
            print(f"[Pipeline] ❌ FATAL ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    
factcheck_pipeline = FactCheckPipeline()
