from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

class HybridRetrieval:
    """
    BM25 + Dense + Entity Boost Retrieval
    """
    def __init__(self):
        # Tái sử dụng model MiniLM gọn nhẹ giống các service khác
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.alpha = 0.4
        self.beta = 0.4
        self.gamma = 0.2

    def _bm25_score(self, query_tokens: List[str], docs_tokens: List[List[str]]) -> np.ndarray:
        if not docs_tokens:
            return np.array([])
        bm25 = BM25Okapi(docs_tokens)
        return np.array(bm25.get_scores(query_tokens))

    def _dense_score(self, query: str, docs: List[str]) -> np.ndarray:
        if not docs:
            return np.array([])
        q_emb = self.dense_model.encode(query, convert_to_tensor=True)
        d_emb = self.dense_model.encode(docs, convert_to_tensor=True)
        return util.cos_sim(q_emb, d_emb)[0].cpu().numpy()

    def _entity_boost(self, entity_set: List[str], doc_titles: List[str]) -> np.ndarray:
        scores = []
        entity_set = set([e.lower() for e in entity_set if e])

        for t in doc_titles:
            if t and t.lower() in entity_set:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return np.array(scores)

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        scores = np.array(scores, dtype=np.float32)
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            # Nếu tất cả điểm bằng nhau, đưa về mảng 0 để tránh lỗi chia cho 0 (division by zero)
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)

    def rank(self, query: str, docs: List[Dict], entity_links: List[str] = None) -> List[tuple]:
        """
        docs: [{title, text}]
        Returns: List of tuples (doc, final_score)
        """
        if not docs:
            return []

        titles = [d["title"] for d in docs]
        texts = [d["text"] for d in docs]

        # Tokenize cơ bản theo khoảng trắng
        tokenized_docs = [t.split() for t in texts]
        query_tokens = query.split()

        bm25_scores = self._bm25_score(query_tokens, tokenized_docs)
        dense_scores = self._dense_score(query, texts)
        entity_scores = self._entity_boost(entity_links or [], titles)

        # Chuẩn hóa khoảng điểm về [0.0, 1.0]
        bm25_scores = self._normalize(bm25_scores)
        dense_scores = self._normalize(dense_scores)

        # Tính điểm gộp trọng số
        final_scores = (
            self.alpha * bm25_scores +
            self.beta * dense_scores +
            self.gamma * entity_scores
        )

        ranked = sorted(
            zip(docs, final_scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked