"""
Script test doc lap tang NLI (RoBERTa) tích hợp chuẩn cấu trúc EntailmentService.
Chạy luồng từ Retrieval -> Selector -> Dự đoán nhãn bằng RoBERTa.
Tối ưu hóa Batching chạy cuốn chiếu trên CPU để tránh treo máy.
"""
import asyncio
import sys
import time
import urllib.parse
from datasets import load_dataset
import traceback

# Import các service độc lập theo đúng kiến trúc của bạn
from app.services.retrieval_service import query_builder, retrieval_service
from app.services.ranking_service import ranking_service, sentence_selector
from app.services.entailment_service import entailment_service

try:
    from app.services.retrieval_service import entity_linker
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    entity_linker = None


def clean_wiki_title(url_or_title: str) -> str:
    if not url_or_title: return ""
    if "wikipedia.org/wiki/" in url_or_title:
        url_or_title = url_or_title.split("wikipedia.org/wiki/")[-1]
    decoded_title = urllib.parse.unquote(url_or_title)
    cleaned = decoded_title.strip().replace(" ", "_").lower()
    cleaned = cleaned.replace("-lrb-", "(").replace("-rrb-", ")")
    return cleaned


def map_roberta_to_fever(roberta_verdict: str) -> str:
    """
    Ánh xạ nhãn từ đầu ra của RoBERTa (CONTRADICTS) 
    về nhãn chuẩn của bộ dữ liệu FEVER (REFUTES)
    """
    v = roberta_verdict.upper().strip()
    if v == "CONTRADICTS":
        return "REFUTES"
    return v


async def process_nli_sample(sample, current_idx, total_count):
    """Xử lý luồng lấy câu bằng chứng và đẩy sang predict_verdict của bạn"""
    claim = sample["claim"]
    gold_label = sample["label"].upper().strip() # SUPPORTS hoặc REFUTES

    try:
        # 1. Trích xuất topic & entity
        topics = query_builder.extract_topics(claim)
        entity_links = []
        if nlp and entity_linker:
            doc = nlp(claim)
            mentions = entity_linker.extract_mentions(doc)
            for m in mentions:
                linked = entity_linker.link(m)
                if linked: entity_links.append(linked)
        
        # 2. Retrieval trang tài liệu
        evidences = retrieval_service.retrieve(topics=topics, entity_links=entity_links)
        if not evidences:
            return None # Bỏ qua nếu không cào được trang

        # 3. Rerank tài liệu & Selector chọn câu bằng chứng
        scored_documents = await ranking_service.rank_evidence(claim, evidences)
        scored_documents = sorted(scored_documents, key=lambda x: x.final_score, reverse=True)
        top_evidences = [e.evidence for e in scored_documents[:5]]
        
        sentence_evidences = sentence_selector.select(
            claim=claim, evidences=top_evidences, top_n=3
        )
        if not sentence_evidences:
            return None

        # 4. Nối các câu bằng chứng thành chuỗi pseudo_outline giống cấu trúc hàm của bạn mong muốn
        pseudo_outline = " ".join([ev.text for ev in sentence_evidences])
        
        # 5. 🔥 GỌI HÀM PREDICT_VERDICT CỦA FILE ENTAILMENT SERVICE
        # Hàm nhận (claim, pseudo_outline) và trả về một dict chứa "verdict"
        res_dict = await entailment_service.predict_verdict(claim=claim, pseudo_outline=pseudo_outline)
        
        pred_raw = res_dict.get("verdict", "NOT_ENOUGH_INFO")
        # Ánh xạ CONTRADICTS -> REFUTES để so khớp công bằng với FEVER
        pred_label = map_roberta_to_fever(pred_raw)
        
        is_correct = (pred_label == gold_label)
        status = "DUNG" if is_correct else " SAI "
        
        print(f" [{current_idx}/{total_count}] {status} | Claim: {claim[:45]}...")
        print(f" Gold FEVER: {gold_label} | RoBERTa Pred: {pred_raw} (Mapped: {pred_label})")
        
        return is_correct

    except Exception as e:
        print(f" [{current_idx}/{total_count}]  Loi luong NLI: {str(e)}")
        traceback.print_exc()
        return False


async def test_nli_accuracy(samples: int = 30, batch_size: int = 2):
    """
    Hàm kiểm thử NLI cuốn chiếu theo Batch nhỏ tối ưu hóa cho CPU.
    Khuyên dùng samples = 30 mẫu để lấy nhanh tỷ lệ chính xác ban đầu.
    """
    print("[INIT] Dang tai bo du lieu FEVER v1.0...")
    dataset = load_dataset("fever", "v1.0")
    dev_set = dataset["labelled_dev"]
    
    # Lọc lấy các mẫu có nhãn phân loại rõ ràng
    valid_samples = [s for s in dev_set if s["label"] in ["SUPPORTS", "REFUTES"]][:samples]
    total_samples = len(valid_samples)
    
    print(f"[START] Kich hoat danh gia RoBERTa NLI tren {total_samples} mau...")
    print(f" Che đo toi uu CPU: Cuon chieu theo cum (Batch size = {batch_size})")
    print("=" * 85)
    
    start_time = time.time()
    nli_hit = 0
    total_valid_executed = 0
    
    # Chạy cuốn chiếu để bảo vệ CPU không bị quá tải 100%
    for i in range(0, total_samples, batch_size):
        batch = valid_samples[i:i + batch_size]
        
        tasks = [
            process_nli_sample(sample, i + idx + 1, total_samples)
            for idx, sample in enumerate(batch)
        ]
        
        batch_results = await asyncio.gather(*tasks)
        
        for res in batch_results:
            if res is not None:
                total_valid_executed += 1
                if res is True:
                    nli_hit += 1
                    
        await asyncio.sleep(0.3) # Giãn cách nhỏ để CPU giải nhiệt
        sys.stdout.flush()

    end_time = time.time()
    print("\n" + "=" * 25 + " ROBERTA NLI ACCURACY REPORT " + "=" * 25)
    print(f"Tong thoi gian chay test         : {end_time - start_time:.2f} giây (~{(end_time - start_time)/60:.1f} phút)")
    print(f"So mau kiem tra hop le thanh cong: {total_valid_executed}")
    print(f"So mau RoBERTa đoan đung nhan    : {nli_hit}")
    if total_valid_executed > 0:
        print(f"DO CHINH XAC NHAN CUOI CUNG (Accuracy): {nli_hit / total_valid_executed * 100:.2f}%")
    print("=" * 79)


if __name__ == "__main__":
    # Đặt mặc định 30 mẫu để CPU quét nhanh tầm vài phút là ra kết quả
    asyncio.run(test_nli_accuracy(samples=30, batch_size=2))