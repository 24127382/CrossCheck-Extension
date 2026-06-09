from typing import List
import re
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from ..schemas.evidence import Evidence, EvidenceScore

class RankingService:
    """Service for ranking and filtering evidence at both Document and Sentence levels"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # FIX: Liên kết và tái sử dụng toàn cục instance sentence_selector cho cross-encoder scoring
        self.sentence_selector = sentence_selector
        
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
        doc_embs = self.model.encode([e.text for e in evidences], convert_to_tensor=True)

        scores = util.cos_sim(claim_emb, doc_embs)[0].tolist()

        ranked = []
        for i, s in enumerate(scores):
            evidences[i].score = s
            
            # FIX: Tích hợp tín hiệu điểm cấp câu bằng Cross-Encoder (SentenceSelector)
            try:
                sent_scores = self.sentence_selector.select(claim, [evidences[i]])
                sentence_score = max([x.score for x in sent_scores]) if sent_scores else 0.0
            except:
                sentence_score = 0.0

            ranked.append(EvidenceScore(
                evidence=evidences[i],
                relevance_score=s,
                sentence_score=sentence_score,
                entailment_score=0.0,
                final_score=s  # updated sau ở aggregation layer (scoring module)
            ))
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
            for s in self.split(e.text):
                candidates.append(s)
                metadata.append(e.source)

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
            Evidence(text=s, source=src, score=float(score))
            for s, src, score in ranked[:top_n]
        ]
        
        
def compute_final_score(relevance, sentence_score, entailment):
    # FIX: Giới hạn (clamp) dữ liệu đầu vào để tránh score bùng nổ vượt ngưỡng chuẩn 0-1
    relevance = max(0.0, min(relevance, 1.0))
    sentence_score = max(0.0, min(sentence_score, 1.0))
    entailment = max(0.0, min(entailment, 1.0))

    # Đảm bảo tính gộp điểm ổn định cho FEVER eval consistency
    return (
        0.3 * relevance +
        0.3 * sentence_score +
        0.4 * entailment
    )

# Lưu ý khởi tạo đúng thứ tự: selector trước để service gán được thuộc tính trong __init__
sentence_selector = SentenceSelector()
ranking_service = RankingService()