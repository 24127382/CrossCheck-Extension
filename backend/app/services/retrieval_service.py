"""Retrieval service - handles evidence retrieval from Wikipedia"""
import spacy
from typing import List, Optional
from ..schemas.evidence import Evidence
from hybrid_retrieval import HybridRetrieval  
from wikipedia_client import WikipediaClient, wiki_client  # Import một chiều từ hạ tầng    
from entity_linker import EntityLinker, entity_linker  # Import một chiều từ hạ tầng
# Load spacy model for NER
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Spacy model not found. Please run: python -m spacy download en_core_web_lg")
    nlp = None

class QueryBuilder:
    def __init__(self, linker: EntityLinker = None):
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except:
            self.nlp = None
        self.linker = linker

    def extract_topics(self, text: str):
        if not self.nlp:
            return [text]

        doc = self.nlp(text)
        linked_entities = []

        if self.linker:
            mentions = self.linker.extract_mentions(doc)
            for m in mentions:
                linked = self.linker.link(m)
                if linked:
                    linked_entities.append(linked)

        noun_chunks = [chunk.text for chunk in doc.noun_chunks if len(chunk.text) > 2]
        candidates = linked_entities + noun_chunks
        candidates.append(text)

        seen = set()
        result = []
        for q in candidates:
            if not q:
                continue
            q_low = q.lower().strip()
            if q_low not in seen and len(q) > 2:
                seen.add(q_low)
                result.append(q)

        return result[:8]


class RetrievalService:
    def __init__(self, client: WikipediaClient, linker: EntityLinker):
        self.client = client
        self.linker = linker
        self.hybrid = HybridRetrieval()  # 🔥 Khởi tạo Hybrid Engine

    def retrieve(self, topics: List[str], entity_links: Optional[List[str]] = None) -> List[Evidence]:
        all_docs = []
        seen = set()

        # Tạo bản sao tránh biến đổi list gốc
        expanded_topics = list(topics)
        if topics:
            expanded_topics.append(" ".join(topics))
            expanded_topics.append(topics[0] + " wikipedia")

        for topic in expanded_topics:
            search = self.client.search(topic)
            results = search.get("query", {}).get("search", [])

            if not results:
                continue

            for r in results[:5]:  # Mở rộng lấy top 5 tài liệu thô ban đầu để rerank
                title = r["title"]
                if title in seen:
                    continue
                seen.add(title)

                extract = self.client.extract(title)
                pages = extract.get("query", {}).get("pages", {})
                if not pages:
                    continue
                page = list(pages.values())[0]

                text = page.get("extract", "")
                if text and len(text) > 100:
                    all_docs.append({
                        "title": title,
                        "text": text
                    })

        # Fallback nếu danh sách trống hoàn toàn trước khi xếp hạng
        if not all_docs and topics:
            fallback = self.client.search(topics[0])
            results = fallback.get("query", {}).get("search", [])
            if results:
                title = results[0]["title"]
                extract = self.client.extract(title)
                pages = extract.get("query", {}).get("pages", {})
                page = list(pages.values())[0]
                text = page.get("extract", "")
                if text:
                    all_docs.append({"title": title, "text": text})

        if not all_docs:
            return []

        # 🔥 TỰ ĐỘNG PHÒNG THỦ: Nếu luồng gọi ngoài chưa truyền entity_links, tự động sinh ra
        if entity_links is None:
            entity_links = []
            combined_query = " ".join(topics)
            if nlp:
                doc = nlp(combined_query)
                mentions = self.linker.extract_mentions(doc)
                for m in mentions:
                    linked = self.linker.link(m)
                    if linked:
                        entity_links.append(linked)
        # --- beam search function here ---
        
        
        # 🔥 HYBRID RANKING CORE ACTIVATED
        query_str = " ".join(topics)
        ranked = self.hybrid.rank(
            query=query_str,
            docs=all_docs,
            entity_links=entity_links
        )

        # Trả về Top 10 thực thể Evidence chất lượng cao nhất sau khi lọc hỗn hợp
        return [
            Evidence(text=d["text"], source=d["title"])
            for d, _ in ranked[:10]
        ]

# Khởi tạo các Service đơn lẻ dùng chung toàn cục
query_builder = QueryBuilder(linker=entity_linker)
retrieval_service = RetrievalService(client=wiki_client, linker=entity_linker)
