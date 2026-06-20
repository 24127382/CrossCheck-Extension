"""Factcheck pipeline - orchestrates the fact-checking flow"""
from typing import Optional
from ..schemas.response import FactCheckResponse, EvidenceSchema
from ..services.retrieval_service import retrieval_service, query_builder, entity_linker, nlp
from ..services.entailment_service import entailment_service
from ..services.ranking_service import ranking_service, sentence_selector
from ..services.cache_service import cache_service


class FactCheckPipeline:
    """Main fact-checking pipeline"""
    
    async def process(self, claim: str, context: Optional[str] = None) -> FactCheckResponse:
        """
        Main fact-checking pipeline: 
        claim → topics & entity links → Hybrid Retrieve (BM25+Dense+Entity) → rank → NLI score → aggregate → filter → RoBERTa verdict → response
        """
        try:
            # Check cache first
            cached_result = cache_service.get(claim)
            if cached_result:
                print(f"[Pipeline] ✓ Using cached result for: {claim[:100]}")
                return FactCheckResponse(
                    claim=claim,
                    verdict=cached_result.verdict,
                    confidence=cached_result.confidence,
                    summary=" ".join([e.text for e in cached_result.evidences]),
                    evidences=[
                        EvidenceSchema(
                            source=e.source,
                            stance=e.stance,
                            score=e.score,
                            text=e.text
                        )
                        for e in cached_result.evidences
                    ]
                )
            
            print(f"\n[Pipeline] Starting fact-check for: {claim[:100]}")
            
            # Step 1: Noun chunking & Topic extraction + Entity Linking
            print("[Pipeline] Step 1: Extracting topics and grounding entity links...")
            topics = query_builder.extract_topics(claim)
            
            # Thu thập danh sách entity_links làm dữ liệu bổ trợ tăng hạng (Boost) cho Hybrid Retrieval
            entity_links = []
            if nlp:
                doc = nlp(claim)
                mentions = entity_linker.extract_mentions(doc)
                for m in mentions:
                    linked = entity_linker.link(m)
                    if linked:
                        entity_links.append(linked)
            
            # Step 2: Tìm kiếm kết hợp thuật toán Hybrid (BM25 + Dense + Entity Boost)
            print(f"[Pipeline] Step 2: Running Hybrid Retrieval for topics: {topics} (Boosted Entities: {entity_links})")
            # 🔥 ĐÃ TÍCH HỢP: Truyền thêm mảng entity_links vào hàm retrieve
            #------------------------------------------------------------------------------
            evidences = retrieval_service.retrieve(topics=topics, entity_links=entity_links)
            print(f"[Pipeline] Retrieved & Hybrid-Ranked {len(evidences)} evidence items")
            
            if not evidences:
                print("[Pipeline] ⚠️ No evidence retrieved!")
                return FactCheckResponse(
                    claim=claim,
                    verdict="NOT_ENOUGH_INFO",
                    confidence=0.0,
                    summary="",
                    evidences=[]
                )
            # Step 3: Đưa qua Cross-Encoder cấp câu để tinh chỉnh điểm tương đồng (Relevance & Sentence similarity)
            print("[Pipeline] Step 3: Fine-ranking evidence via Cross-Encoder...")
            scored_documents = await ranking_service.rank_evidence(claim, evidences)
            
            # Sắp xếp lại danh sách theo điểm final_score thực tế sau gộp
            scored_documents = sorted(scored_documents, key=lambda x: x.final_score, reverse=True)

            if not scored_documents:
                return FactCheckResponse(
                    claim=claim,
                    verdict="NOT_ENOUGH_INFO",
                    confidence=0.0,
                    summary="",
                    evidences=[]
                )
                    
            # Step 4: Chọn top-k documents tốt nhất để tiến hành trích xuất câu ngắn bằng chứng
            print("[Pipeline] Step 4: Selecting top-k documents...")
            top_evidences = [e.evidence for e in scored_documents[:10]]
            
            # Step 5: Tách câu và lấy câu liên quan nhất (Cross-Encoder rerank cấp câu)
            print("[Pipeline] Step 5: Extracting and ranking sentences...")
            sentence_evidences  = sentence_selector.select(
                claim=claim,
                evidences=top_evidences,
                top_n=2
            )
            
            # Nếu không trích xuất được câu nào, fallback trả lỗi
            if not sentence_evidences :
                print("[Pipeline] ⚠️ No clean sentences extracted!")
                return FactCheckResponse(
                    claim=claim,
                    verdict="NOT_ENOUGH_INFO",
                    confidence=0.0,
                    summary="",
                    evidences=[]
                )

            pseudo_outline = [
                ev.text
                for ev in sentence_evidences
            ]
            
            # Step 6: Chạy RoBERTa Predict Verdict dựa trên câu bằng chứng cô đọng tốt nhất
            print("[Pipeline] Step 6: Running RoBERTa prediction on pure sentences...")
            verdict_result = await entailment_service.predict_verdict(
                claim=claim,
                evidences=pseudo_outline
            )
            print(f"[Pipeline]  RoBERTa verdict: {verdict_result['verdict']} (confidence: {verdict_result['confidence']:.3f})")
            
            # Step 7: Format response thành schema đầu ra
            print("[Pipeline] Step 7: Formatting response...")

            verdict = verdict_result["verdict"]
            confidence = verdict_result["confidence"]

            if verdict in ["SUPPORTS", "REFUTES"] and confidence < 0.70:
                verdict = "NOT_ENOUGH_INFO"
            
            response = FactCheckResponse(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                summary=" ".join(pseudo_outline),
                evidences=[
                    EvidenceSchema(
                        source=e.source,
                        stance=e.stance,
                        score=e.score,
                        text=e.text
                    )
                    for e in sentence_evidences
                ]
            )
            
            print("[Pipeline] Step 8: Pipeline execution finish.")
            print("[Pipeline]  Pipeline complete!")
            # Save to cache
            cache_service.save(
                claim=claim,
                verdict=response.verdict,
                confidence=response.confidence,
                evidences=response.evidences
            )

            return response
            
        except Exception as e:
            print(f"[Pipeline] ❌ FATAL ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    
    
factcheck_pipeline = FactCheckPipeline()