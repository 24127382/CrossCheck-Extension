from typing import List
import re
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from ..schemas.evidence import Evidence, EvidenceScore

class RankingService:
    """Service for ranking and filtering evidence at both Document and Sentence levels"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # FIX: Liên kết và tái sử dụng toàn cục instance sentence_selector cho cross-encoder scoring
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Tách câu thông minh hơn dùng Regex, hạn chế lỗi khi gặp U.S., $1.5, ..."""
        # Tách theo dấu chấm, chấm hỏi, chấm than nếu có khoảng trắng theo sau
        sentences = re.split(r'(?<=[.!?])\s+', text)
        cleaned = []
        for s in sentences:
            s_clean = s.strip()
            # Chỉ giữ câu có độ dài hợp lý (bỏ qua câu quá ngắn hoặc rác văn bản)
            if len(s_clean) > 15:
                cleaned.append(s_clean)
        return cleaned

    async def rank_evidence(self, claim: str, evidences: List[Evidence]) -> List[EvidenceScore]:
        claim_emb = self.model.encode(claim, convert_to_tensor=True)
        doc_embs = self.model.encode([e.text[:1500] for e in evidences], convert_to_tensor=True)

        scores = util.cos_sim(claim_emb, doc_embs)[0].tolist()

        ranked = []
        for i, s in enumerate(scores):
            evidences[i].score = s
            
            ranked.append(
            EvidenceScore(
                evidence=evidences[i],
                relevance_score=s,
                entailment_score=0.0,
                final_score=s
            )
        )
        return sorted(ranked, key=lambda x: x.final_score, reverse=True)

class SentenceSelector:
    def __init__(self):
        # Cross-encoder reranker (lightweight, production friendly)
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def split(self, text: str):
        return [
            s.strip()
            for s in re.split(r'(?<=[.!?])\s+', text)
            if len(s.strip()) > 15
        ]

    def select(self, claim: str, evidences: List[Evidence], top_n=3):
        candidates = []
        metadata = []

        # 1. collect sentences
        for e in evidences:
            for idx, s in enumerate(self.split(e.text)):
                candidates.append(s)
                metadata.append({
                    "source": e.source,
                    "sentence_id": idx
                })

        if not candidates:
            return []

        # 2. build cross-encoder pairs
        pairs = [(claim, sent) for sent in candidates]

        # 3. predict relevance scores
        scores = self.model.predict(pairs)

        # 4. rank
        ranked = sorted(
            zip(candidates, metadata, scores),
            key=lambda x: x[2],
            reverse=True
        )

        # 5. return top-N
        return [
            Evidence(
                text=sent,
                source=meta["source"],
                sentence_id=meta["sentence_id"],
                score=float(score)
            )
            for sent, meta, score in ranked[:top_n]
        ]
                
        

# Lưu ý khởi tạo đúng thứ tự: selector trước để service gán được thuộc tính trong __init__
sentence_selector = SentenceSelector()
ranking_service = RankingService()