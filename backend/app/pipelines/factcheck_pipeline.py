"""Factcheck pipeline - orchestrates the fact-checking flow"""
from typing import Optional
from ..schemas.response import FactCheckResponse, EvidenceSchema
from ..services.retrieval_service import retrieval_service, query_builder, entity_linker, nlp
from ..services.entailment_service import entailment_service
from ..services.ranking_service import ranking_service, sentence_selector, compute_final_score
from ..services.cache_service import cache_service
from ..schemas.evidence import Evidence

class FactCheckPipeline:
    """Main fact-checking pipeline"""
    
    async def process(self, claim: str, context: Optional[str] = None) -> FactCheckResponse:
        """
        Main fact-checking pipeline: 
        claim → topics & entity links → Hybrid Retrieve (BM25+Dense+Entity) → rank → NLI score → aggregate → filter → RoBERTa verdict → response
        """
        try:
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
            evidences = retrieval_service.retrieve(topics=topics, entity_links=entity_links)
            print(f"[Pipeline] ✅ Retrieved & Hybrid-Ranked {len(evidences)} evidence items")
            
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
            
            # Step 3.5: Tính điểm Entailment từ mô hình RoBERTa-MNLI
            print("[Pipeline] Step 3.5: Computing entailment scores for documents...")
            docs_to_entail = [sd.evidence for sd in scored_documents]
            entailment_results = await entailment_service.compute_entailment(claim, docs_to_entail)
            
            # Ánh xạ điểm Entailment ngược lại vào danh sách scored_documents và tính điểm Final tổng hợp
            for sd, er in zip(scored_documents, entailment_results):
                sd.entailment_score = er.entailment_score
                # Áp dụng hàm gộp điểm kết hợp clamping [0.0, 1.0]
                sd.final_score = compute_final_score(
                    relevance=sd.relevance_score,
                    sentence_score=sd.sentence_score,
                    entailment=sd.entailment_score
                )
            
            # Sắp xếp lại danh sách theo điểm final_score thực tế sau gộp
            scored_documents = sorted(scored_documents, key=lambda x: x.final_score, reverse=True)

            # Step 4: Lọc phần tử theo ngưỡng threshold điểm gộp mới (final_score >= 0.3)
            print("[Pipeline] Step 4: Filtering evidence based on final score threshold...")
            filtered_evidences = [e for e in scored_documents if e.final_score >= 0.3]
            
            if not filtered_evidences and scored_documents:
                print("[Pipeline] ⚠️ Threshold quá cao! Lấy tạm Top 2 kết quả tốt nhất...")
                filtered_evidences = scored_documents[:2]
            
            print(f"[Pipeline] ✅ Filtered: {len(scored_documents)} → {len(filtered_evidences)}")
            
            if not filtered_evidences:
                print("[Pipeline] ⚠️ No evidence passed threshold!")
                return FactCheckResponse(
                    claim=claim,
                    verdict="NOT_ENOUGH_INFO",
                    confidence=0.0,
                    summary="",
                    evidences=[]
                )
                
            # Step 5: Chọn top-k documents tốt nhất để tiến hành trích xuất câu ngắn bằng chứng
            print("[Pipeline] Step 5: Selecting top-k documents...")
            top_evidences = [e.evidence for e in filtered_evidences[:5]]
            
            # Step 6: Tách câu và lấy câu liên quan nhất (Cross-Encoder rerank cấp câu)
            print("[Pipeline] Step 6: Extracting and ranking sentences...")
            clean_sentence_evidences = sentence_selector.select(
                claim=claim,
                evidences=top_evidences,
                top_n=1
            )
            
            # Nếu không trích xuất được câu nào, fallback trả lỗi
            if not clean_sentence_evidences:
                print("[Pipeline] ⚠️ No clean sentences extracted!")
                return FactCheckResponse(
                    claim=claim,
                    verdict="NOT_ENOUGH_INFO",
                    confidence=0.0,
                    summary="",
                    evidences=[]
                )

            best_sentence = clean_sentence_evidences[0].text
            
            # Step 7: Chạy RoBERTa Predict Verdict dựa trên câu bằng chứng cô đọng tốt nhất
            print("[Pipeline] Step 7: Running RoBERTa prediction on pure sentences...")
            verdict_result = await entailment_service.predict_verdict(
                claim=claim,
                pseudo_outline=best_sentence
            )
            print(f"[Pipeline] ✅ RoBERTa verdict: {verdict_result['verdict']} (confidence: {verdict_result['confidence']:.3f})")
            
            # Step 8: Format response thành schema đầu ra
            print("[Pipeline] Step 8: Formatting response...")
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
            
            print("[Pipeline] Step 9: Pipeline execution finish.")
            print("[Pipeline] ✅ Pipeline complete!")
            return response
            
        except Exception as e:
            print(f"[Pipeline] ❌ FATAL ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

factcheck_pipeline = FactCheckPipeline()