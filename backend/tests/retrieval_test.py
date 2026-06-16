"""
Script danh gia hieu nang thuat toan Hybrid Retrieval (BM25 + Dense + Entity)
tren bo du lieu FEVER v1.0.
"""
import time
import urllib.parse
import sys
from datasets import load_dataset

# Dam bao import dung cau truc thu muc cua ban
from app.services.retrieval_service import retrieval_service, query_builder, entity_linker, nlp


def clean_wiki_title(url_or_title: str) -> str:
    if not url_or_title:
        return ""
    if "wikipedia.org/wiki/" in url_or_title:
        url_or_title = url_or_title.split("wikipedia.org/wiki/")[-1]
    
    decoded_title = urllib.parse.unquote(url_or_title)
    cleaned = decoded_title.strip().replace(" ", "_").lower()
    
    # Dong bo hoa dinh dang dau ngoac cua nhan vang FEVER sang ngoac tieu chuan
    cleaned = cleaned.replace("-lrb-", "(").replace("-rrb-", ")")
    
    return cleaned


def extract_gold_pages(example) -> set:
    pages = set()
    wiki_url_data = example.get("evidence_wiki_url")
    
    if not wiki_url_data:
        return pages

    def recurse_extract(data):
        if isinstance(data, list):
            for item in data:
                recurse_extract(item)
        elif isinstance(data, str) and data.strip():
            cleaned = clean_wiki_title(data)
            if cleaned:
                pages.add(cleaned)

    recurse_extract(wiki_url_data)
    return pages


def evaluate_hybrid_retrieval(samples: int = 100):
    print("[INIT] Dang tai bo du lieu FEVER v1.0 tu Hugging Face...")
    try:
        dataset = load_dataset("fever", "v1.0")
        dev_set = dataset["labelled_dev"]
        print(f"[SUCCESS] Tai xong dataset. Tong so mau dev: {len(dev_set)}")
    except Exception as e:
        print(f"[ERROR] Khong the tai dataset: {str(e)}")
        return

    total_valid_claims = 0
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_rank_sum = 0

    print(f"[START] Bat dau danh gia HYBRID RETRIEVAL tren {samples} mau...")
    print("=" * 80)

    try:
        for idx in range(min(samples, len(dev_set))):
            sample = dev_set[idx]
            claim = sample["claim"]
            gold_pages = extract_gold_pages(sample)

            if not gold_pages:
                continue

            print(f"\n[{idx}] PROCESSING CLAIM: {claim}")
            print(f" |-- NHAN VANG: {list(gold_pages)}")

            # Giam tan suat goi mang de tranh bi block IP tu Wikipedia
            time.sleep(0.3)

            try:
                # 1. Trích xuất Topics
                topics = query_builder.extract_topics(claim)
                
                # 2. Trích xuất Entity Links (Mô phỏng giống hệt FactCheckPipeline)
                entity_links = []
                if nlp:
                    doc = nlp(claim)
                    mentions = entity_linker.extract_mentions(doc)
                    for m in mentions:
                        linked = entity_linker.link(m)
                        if linked:
                            entity_links.append(linked)

                print(f"  |-- Topics: {topics}")
                print(f"  |-- Entity Links: {entity_links}")
                
                # 3. Kích hoạt Hybrid Retrieval
                print(" |-- [STEP] Dang chay Hybrid Retrieval (BM25 + Dense + Entity Boost)...")
                evidences = retrieval_service.retrieve(topics=topics, entity_links=entity_links)
                
                # 4. Chuan hoa ket qua dau ra
                # Lưu ý: Hybrid trả về List[Evidence], ta truy cập thuộc tính .source
                retrieved_pages = []
                for ev in evidences:
                    cleaned_title = clean_wiki_title(ev.source)
                    if cleaned_title and cleaned_title not in retrieved_pages:
                        retrieved_pages.append(cleaned_title)

                total_valid_claims += 1
                rank = None

                # 5. Tinh toan xep hang
                for i, page in enumerate(retrieved_pages):
                    if page in gold_pages:
                        rank = i + 1
                        break

                if rank == 1:
                    hit_at_1 += 1
                if rank is not None and rank <= 3:
                    hit_at_3 += 1
                if rank is not None and rank <= 5:
                    hit_at_5 += 1
                if rank is not None:
                    reciprocal_rank_sum += 1 / rank

                # Log ket qua
                print(f" |-- KET QUA HYBRID: {retrieved_pages[:5]}")
                print(f" |-- THU HANG RANK: {rank if rank is not None else 'N/A'}")
                print("-" * 60)
                
                sys.stdout.flush()

            except Exception as e:
                print(f" [ERROR] Loi logic xu ly tai mau [{idx}]: {str(e)}")

    except KeyboardInterrupt:
        print("\n[WARNING] Tien trinh bi dung boi nguoi dung (Ctrl+C). Dang xuat ket qua...")

    finally:
        if total_valid_claims == 0:
            print("\n========== KET QUA DANH GIA ==========")
            print("Khong co mau hop le nao duoc xu ly thanh cong.")
            print("======================================")
            return

        print("\n========== FINAL HYBRID RETRIEVAL RESULT ==========")
        print(f"Tong so Claims hop le thuc te : {total_valid_claims}")
        print(f"Recall@1                       : {hit_at_1 / total_valid_claims:.4f}")
        print(f"Recall@3                       : {hit_at_3 / total_valid_claims:.4f}")
        print(f"Recall@5                       : {hit_at_5 / total_valid_claims:.4f}")
        print(f"MRR (Mean Reciprocal Rank)     : {reciprocal_rank_sum / total_valid_claims:.4f}")
        print("===================================================")


if __name__ == "__main__":
    evaluate_hybrid_retrieval(samples=100)