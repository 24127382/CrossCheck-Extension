"""Retrieval service - handles evidence retrieval from Wikipedia"""
import heapq
import urllib.parse
import spacy
from typing import List, Dict, Optional
from sentence_transformers import util

from ..schemas.evidence import Evidence, Node
from app.services.hybrid_retrieval import HybridRetrieval  
from app.services.wikipedia_client import WikipediaClient, wiki_client     
from app.services.entity_linker import EntityLinker, entity_linker  

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
        self.hybrid = HybridRetrieval()  

    def retrieve(self, topics: List[str], entity_links: Optional[List[str]] = None) -> List[Evidence]:
        all_docs = []
        seen = set()

        expanded_topics = list(topics)
        if topics:
            expanded_topics.append(" ".join(topics))
            expanded_topics.append(topics[0] + " wikipedia")

        for topic in expanded_topics:
            search = self.client.search(topic)
            results = search.get("query", {}).get("search", [])

            if not results:
                continue

            for r in results[:5]:  
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
        
        query_str = " ".join(topics)
        ranked = self.hybrid.rank(
            query=query_str,
            docs=all_docs,
            entity_links=entity_links
        )

        return [
            Evidence(text=d["text"], source=d["title"])
            for d, _ in ranked[:10]
        ]

# Khởi tạo các Service đơn lẻ dùng chung toàn cục
query_builder = QueryBuilder(linker=entity_linker)
retrieval_service = RetrievalService(client=wiki_client, linker=entity_linker)


# ==========================================
# GRAPH RETRIEVAL SERVICE ENGINE
# ==========================================
class GraphRetrievalService:
    """
    Ultra-Optimized Beam Search Graph Retrieval
    Đã fix: Pre-compute Claim Embeddings, Limit Entity Scope, Refined Score.
    """

    def __init__(self, client, linker, max_depth=2, beam_size=3):
        self.client = client
        self.linker = linker
        self.max_depth = max_depth
        self.beam_size = beam_size
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Cache lưu kết quả API
        self._page_cache = {}

    @staticmethod  
    def clean_wiki_title(url_or_title: str) -> str:  
        if not url_or_title:
            return ""
        if "wikipedia.org/wiki/" in url_or_title:
            url_or_title = url_or_title.split("wikipedia.org/wiki/")[-1]
        decoded_title = urllib.parse.unquote(url_or_title)
        cleaned = decoded_title.strip().replace(" ", "_").lower()
        cleaned = cleaned.replace("-lrb-", "(").replace("-rrb-", ")")
        return cleaned

    def _cached_extract(self, title: str) -> str:
        cleaned_key = self.clean_wiki_title(title)
        if cleaned_key in self._page_cache:
            return self._page_cache[cleaned_key]
        
        try:
            page = self.client.extract(title)
            pages = page.get("query", {}).get("pages", {})
            if not pages: return ""
            data = list(pages.values())[0]
            text = data.get("extract", "")
            self._page_cache[cleaned_key] = text
            return text
        except Exception:
            return ""

    # 🔥 FIX 1 & 4: Nhận thẳng claim_emb và claim_tokens đã tính sẵn từ bên ngoài
    def _score(self, claim_emb, claim_tokens: set, text: str, title: str) -> float:
        try:
            # Chỉ encode text, không encode lại claim
            text_emb = self.model.encode(text[:1000], convert_to_tensor=True)
            dense = float(util.cos_sim(claim_emb, text_emb)[0][0])
        except Exception:
            dense = 0.0

        title_cleaned = title.replace("_", " ").lower()
        title_tokens = set(title_cleaned.split())
        
        # Chỉ dùng overlap như một điểm cộng nhỏ (tie-breaker), không cho phép nó lấn át
        entity_overlap = len(claim_tokens & title_tokens) / (len(title_tokens) + 1e-6)

        # Trọng số ưu tiên tuyệt đối vào Semantic Context thay vì Lexical Match
        return 0.85 * dense + 0.15 * entity_overlap
    
    # 🔥 FIX 3: Ngăn chặn Semantic Drift
    def _expand(self, node: Node, claim_emb, claim_tokens) -> List[Node]:
        children = []
        if node.depth >= self.max_depth:
            return children

        global nlp
        if nlp is None:
            try: nlp = spacy.load("en_core_web_lg")
            except: return children

        # CHỈ ĐỌC 400 KÝ TỰ ĐẦU CỦA NODE: Bắt chính xác anchor entity, loại bỏ rác (Harvard, New York)
        doc = nlp(node.text[:400])
        linked_titles = self.linker.extract_mentions(doc)
        linked_titles = [self.linker.link(m) for m in linked_titles if m]

        seen_local = set()
        unique_titles = []
        for t in linked_titles:
            c_t = self.clean_wiki_title(t)
            if c_t and c_t not in seen_local:
                seen_local.add(c_t)
                unique_titles.append(t)

        for title in unique_titles[:self.beam_size]:
            try:
                text = self._cached_extract(title)
                if len(text) < 100:
                    continue

                # Truyền claim_emb vào _score
                score = self._score(claim_emb, claim_tokens, text, title)
                children.append(Node(
                    title=title, text=text, depth=node.depth + 1,
                    score=score, parent=node.title
                ))
            except Exception:
                continue
        return children

    def retrieve(self, topics: List[str], claim: str) -> List[Dict]:
        frontier = []
        visited = set()
        results = []
        count = 0
        
        # 🔥 FIX 1: TÍNH TOÁN CLAIM ĐÚNG 1 LẦN DUY NHẤT Ở ĐÂY
        # Tính embedding và token set của claim một lần duy nhất để tái sử dụng xuyên suốt quá trình
        claim_emb = self.model.encode(claim, convert_to_tensor=True)
        claim_tokens = set(claim.lower().split())
        
        search_queries = []
        if len(topics) >= 2:
            search_queries.append(f'"{topics[0]} {topics[1]}"')
        search_queries.extend(topics[:3])
        search_queries.append(" ".join(topics))
        
        candidates_seen = set()
        for query in search_queries:
            try:
                search = self.client.search(query)
                for c in search.get("query", {}).get("search", [])[:3]:
                    c_title = self.clean_wiki_title(c["title"])
                    if c_title not in candidates_seen:
                        candidates_seen.add(c_title)
                        
                        text = self._cached_extract(c["title"])
                        if len(text) < 100: continue
                        
                        # Truyền claim_emb vào
                        score = self._score(claim_emb, claim_tokens, text, c["title"])
                        node = Node(c["title"], text, 0, score)
                        heapq.heappush(frontier, (-score, count, node))
                        count += 1
            except Exception:
                continue

        # BEAM SEARCH LOOP
        while frontier and len(results) < self.beam_size:
            _, _, node = heapq.heappop(frontier)
            cleaned_title = GraphRetrievalService.clean_wiki_title(node.title)

            if cleaned_title in visited: continue

            visited.add(cleaned_title)
            results.append(node)

            # Truyền claim_emb đi tiếp
            children = self._expand(node, claim_emb, claim_tokens)
            for child in children:
                cleaned_child_title = GraphRetrievalService.clean_wiki_title(child.title)
                if cleaned_child_title in visited: continue

                heapq.heappush(frontier, (-child.score, count, child))
                count += 1

        return [
            {"title": n.title, "text": n.text, "score": n.score}
            for n in results
        ]