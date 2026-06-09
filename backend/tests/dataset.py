import urllib.parse
from datasets import load_dataset
# Đảm bảo import đúng package từ cấu trúc dự án của bạn
from app.services.retrieval_service import query_builder, retrieval_service, entity_linker, nlp


def clean_wiki_title(url_or_title: str) -> str:
    """
    Chuẩn hóa chính xác tiêu đề từ FEVER dataset và Wikipedia Client về cùng một dạng.
    Ví dụ: 'https://en.wikipedia.org/wiki/Roman_Empire' -> 'Roman_Empire'
           'Fox_Broadcasting_Company' -> 'Fox_Broadcasting_Company'
           'Michael_Jackson_%28singer%29' -> 'Michael_Jackson_(singer)'
    """
    if not url_or_title:
        return ""
    
    # 1. Nếu là URL đầy đủ, chỉ cắt lấy phần tên trang sau cùng
    if "wikipedia.org/wiki/" in url_or_title:
        url_or_title = url_or_title.split("wikipedia.org/wiki/")[-1]
    
    # 2. Decode các ký tự đặc biệt như %20 (khoảng trắng), %28, %29 (dấu ngoặc)
    decoded_title = urllib.parse.unquote(url_or_title)
    
    # 3. Đồng bộ hóa khoảng trắng thành dấu gạch dưới (_)
    return decoded_title.strip().replace(" ", "_")


def extract_gold_pages(example):
    """
    Trích xuất gold page dựa trên kiến trúc thực tế của Hugging Face FEVER v1.0,
    phòng thủ chặt chẽ trước dữ liệu rỗng và lồng mảng dữ liệu phức tạp.
    """
    pages = set()
    wiki_url_data = example.get("evidence_wiki_url")
    
    if not wiki_url_data:
        return pages

    # Đệ quy phá vỡ mọi cấu trúc mảng lồng nhau (Nested lists) trong dữ liệu FEVER
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


import time  # Thêm thư viện để tạo khoảng nghỉ

def evaluate_recall_mrr(samples=200):
    print("Loading FEVER v1.0 dataset from Hugging Face...")
    dataset = load_dataset("fever", "v1.0")
    dev = dataset["labelled_dev"]

    total = 0
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_rank_sum = 0

    print(f"Starting evaluation on {samples} samples...\n")

    try:
        for idx in range(min(samples, len(dev))):
            sample = dev[idx]
            claim = sample["claim"]
            gold_pages = extract_gold_pages(sample)

            if not gold_pages:
                continue

            # ⏳ Giảm tải cho API: Nghỉ một chút trước khi gửi request mới để tránh bị chặn
            time.sleep(0.2)

            try:
                # 1. Trích xuất Topics
                topics = query_builder.extract_topics(claim)
                
                # 2. Tìm kiếm (Lưu ý: Đảm bảo các hàm requests trong client đã có timeout tốt)
                entity_links = []
                if nlp:
                    doc = nlp(claim)
                    mentions = entity_linker.extract_mentions(doc)
                    for m in mentions:
                        linked = entity_linker.link(m)
                        if linked:
                            entity_links.append(linked)
                
                evidences = retrieval_service.retrieve(topics=topics, entity_links=entity_links)

                # 3. Chuẩn hóa đối chứng
                retrieved_pages = []
                for e in evidences:
                    if e:
                        source_attr = getattr(e, "source", "")
                        cleaned_source = clean_wiki_title(source_attr)
                        if cleaned_source:
                            retrieved_pages.append(cleaned_source)

                total += 1
                rank = None

                for i, page in enumerate(retrieved_pages):
                    if page in gold_pages:
                        rank = i + 1
                        break

                if rank == 1: hit_at_1 += 1
                if rank is not None and rank <= 3: hit_at_3 += 1
                if rank is not None and rank <= 5: hit_at_5 += 1
                if rank is not None: reciprocal_rank_sum += 1 / rank

                # IN LOG
                print(f"[{idx}] CLAIM: {claim}")
                print(f" |-- TOPICS EXTRACTED : {topics}")
                print(f" |-- GOLD PAGES       : {list(gold_pages)}")
                print(f" |-- RETRIEVED PAGES  : {retrieved_pages[:5]}")
                print(f" |-- RANK DETERMINED  : {rank}")
                print("-" * 80)

            except Exception as e:
                print(f"❌ ERROR processing sample {idx}: {str(e)}")

    except KeyboardInterrupt:
        # ⚡ PHÒNG THỦ: Nếu bạn bấm Ctrl + C giữa chừng, code sẽ nhảy thẳng xuống đây để in kết quả luôn
        print("\n⚠️ Testing interrupted by user (Ctrl+C). Calculating partial results...")

    finally:
        # 🏁 LUÔN LUÔN IN KẾT QUẢ: Dù bị lỗi mạng hay chạy xong, bảng điểm vẫn sẽ xuất hiện
        if total == 0:
            print("\n========== RESULT ==========")
            print("Processed Samples: 0. Không có dữ liệu để tính toán.")
            print("============================")
            return

        print("\n========== FINAL EVALUATION RESULT ==========")
        print(f"Total Evaluated Claims : {total}")
        print(f"Recall@1               : {hit_at_1 / total:.4f}")
        print(f"Recall@3               : {hit_at_3 / total:.4f}")
        print(f"Recall@5               : {hit_at_5 / total:.4f}")
        print(f"MRR (Mean Reciprocal)  : {reciprocal_rank_sum / total:.4f}")
        print("=============================================")


if __name__ == "__main__":
    # Đặt giới hạn 200 mẫu để thử nghiệm nhanh hiệu năng của thuật toán Hybrid
    evaluate_recall_mrr(samples=200)