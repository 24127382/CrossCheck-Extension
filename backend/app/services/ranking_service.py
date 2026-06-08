from typing import List
import re
from sentence_transformers import SentenceTransformer, util
from ..schemas.evidence import Evidence, EvidenceScore

class RankingService:
    """Service for ranking and filtering evidence at both Document and Sentence levels"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
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
            ranked.append(EvidenceScore(
                evidence=evidences[i],
                relevance_score=s,
                entailment_score=0.0,  # Placeholder, sẽ tính sau
                final_score=s  # Hiện tại chỉ dùng relevance score để xếp hạng
            ))
        return sorted(ranked, key=lambda x: x.final_score, reverse=True)

class SentenceSelector:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def split(self, text: str):
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s) > 15]

    def select(self, evidences: list[Evidence], top_n=1):
        scored = []
        all_sentences = []
        sentence_metadata = []
        
        # Thu thập tất cả câu
        for e in evidences:
            for s in self.split(e.text):
                all_sentences.append(s)
                sentence_metadata.append((s, e.source))
        
        # Batch encode cùng lúc - DON'T convert to tensor, use numpy arrays instead
        embeddings = self.model.encode(all_sentences, convert_to_tensor=False)
        
        # Ghép kết quả - scores are now numpy arrays/floats, not tensors
        scored = [(s, meta[1], float(score.sum()) if hasattr(score, 'sum') else score) 
                for s, meta, score in zip(all_sentences, sentence_metadata, embeddings)]
        
        scored = sorted(scored, key=lambda x: x[2], reverse=True)
        return [Evidence(text=s[0], source=s[1]) for s in scored[:top_n]]

ranking_service = RankingService()
sentence_selector = SentenceSelector()