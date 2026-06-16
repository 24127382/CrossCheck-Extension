"""Factcheck pipeline - orchestrates the fact-checking flow"""
from typing import Optional
from ..schemas.response import FactCheckResponse, EvidenceSchema
from ..services.retrieval_service import retrieval_service, query_builder, entity_linker, nlp
from ..services.entailment_service import entailment_service
from ..services.ranking_service import ranking_service, sentence_selector
from ..services.cache_service import cache_service
from ..schemas.evidence import Evidence
import time
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
            #------------------------------------------------------------------------------
            # Chưa sủ dụng GraphRetrievalService
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
            top_evidences = [e.evidence for e in scored_documents[:5]]
            
            # Step 5: Tách câu và lấy câu liên quan nhất (Cross-Encoder rerank cấp câu)
            print("[Pipeline] Step 5: Extracting and ranking sentences...")
            sentence_evidences  = sentence_selector.select(
                claim=claim,
                evidences=top_evidences,
                top_n=3
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

            best_context = " ".join(
                e.text for e in sentence_evidences
            )
            
            # Step 6: Chạy RoBERTa Predict Verdict dựa trên câu bằng chứng cô đọng tốt nhất
            print("[Pipeline] Step 6: Running RoBERTa prediction on pure sentences...")
            verdict_result = await entailment_service.predict_verdict(
                claim=claim,
                pseudo_outline=best_context
            )
            print(f"[Pipeline]  RoBERTa verdict: {verdict_result['verdict']} (confidence: {verdict_result['confidence']:.3f})")
            
            # Step 7: Format response thành schema đầu ra
            print("[Pipeline] Step 7: Formatting response...")
            response = FactCheckResponse(
                claim=claim,
                verdict=verdict_result.get("verdict", "NOT_ENOUGH_INFO"),
                confidence=verdict_result.get("confidence", 0.0),
                summary=best_context,
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
            return response
            
        except Exception as e:
            print(f"[Pipeline] ❌ FATAL ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    async def process_eval(self, claim: str) -> tuple[list, dict]:
        """
        Phương thức chuyên biệt dành cho Evaluation (Đo đạc hiệu năng).
        - Giữ nguyên y hệt pipeline từ Step 1 đến Step 5.
        - Cắt bỏ hoàn toàn Step 6 (RoBERTa NLI Verdict Prediction) để giảm latency khi test.
        - Trả về tuple: (list_sentence_evidences, debug_info)
        """
        debug_info = {
            "times": {},
            "retrieved_pages": []
        }
        start_pipeline = time.time()
        try:
            # Step 1: Noun chunking & Topic extraction + Entity Linking
            t0 = time.time()
            topics = query_builder.extract_topics(claim)
            
            entity_links = []
            if nlp:
                doc = nlp(claim)
                mentions = entity_linker.extract_mentions(doc)
                for m in mentions:
                    linked = entity_linker.link(m)
                    if linked:
                        entity_links.append(linked)
            debug_info["times"]["step1_topics"] = time.time() - t0
            
            # Step 2: Tìm kiếm kết hợp thuật toán Hybrid (BM25 + Dense + Entity Boost)
            t1 = time.time()
            evidences = retrieval_service.retrieve(topics=topics, entity_links=entity_links)
            debug_info["times"]["step2_retrieval"] = time.time() - t1
            
            # Thu thập danh sách trang tìm kiếm được (Page Retrieval) để tính Page Recall
            for ev in evidences:
                if ev.source not in debug_info["retrieved_pages"]:
                    debug_info["retrieved_pages"].append(ev.source)
            
            if not evidences:
                debug_info["times"]["total_eval_pipeline"] = time.time() - start_pipeline
                return [], debug_info
                
            # Step 3: Đưa qua Cross-Encoder cấp câu để tinh chỉnh điểm tương đồng
            t2 = time.time()
            scored_documents = await ranking_service.rank_evidence(claim, evidences)
            scored_documents = sorted(scored_documents, key=lambda x: x.final_score, reverse=True)
            debug_info["times"]["step3_cross_encoder"] = time.time() - t2

            if not scored_documents:
                debug_info["times"]["total_eval_pipeline"] = time.time() - start_pipeline
                return [], debug_info
                    
            # Step 4: Chọn top-k documents tốt nhất
            t3 = time.time()
            top_evidences = [e.evidence for e in scored_documents[:5]]
            debug_info["times"]["step4_top_k"] = time.time() - t3
            
            # Step 5: Tách câu và lấy câu liên quan nhất (Sentence Selector)
            t4 = time.time()
            # 💡 ĐẢM BẢO: Trong hàm sentence_selector.select của bạn, khi khởi tạo các object Evidence 
            # để trả về, hãy gán: e.sentence_id = <vị trí index của câu trong document gốc>
            sentence_evidences = sentence_selector.select(
                claim=claim,
                evidences=top_evidences,
                top_n=3
            )
            debug_info["times"]["step5_sentence_selector"] = time.time() - t4
            
            # Ghi nhận tổng thời gian chạy nhánh evaluation
            debug_info["times"]["total_eval_pipeline"] = time.time() - start_pipeline
            
            return sentence_evidences, debug_info
            
        except Exception as e:
            print(f"[Pipeline Eval] ❌ ERROR: {str(e)}")
            debug_info["times"]["total_eval_pipeline"] = time.time() - start_pipeline
            raise
    
factcheck_pipeline = FactCheckPipeline()