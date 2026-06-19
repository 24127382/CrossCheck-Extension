from typing import List
import re
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ..schemas.evidence import Evidence, EvidenceScore

class RankingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def rank_evidence(self, claim: str, evidences: List[Evidence]) -> List[EvidenceScore]:
        claim_emb = self.model.encode(claim, convert_to_tensor=True)
        doc_embs = self.model.encode([e.text[:1500] for e in evidences], convert_to_tensor=True)
        scores = util.cos_sim(claim_emb, doc_embs)[0].tolist()
        ranked = []
        for i, s in enumerate(scores):
            evidences[i].score = s
            ranked.append(EvidenceScore(evidence=evidences[i], relevance_score=s, entailment_score=0.0, final_score=s))
        return sorted(ranked, key=lambda x: x.final_score, reverse=True)


class SentenceSelector:
    def __init__(self):
        print("[Selector] Khoi tao Selector 2 tang...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # NLI lần 1 để lọc câu đáng giữ
        repo_id = "Akiya-Vyre/deberta-v3-fever"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        self.nli_model.to(self.device)
        self.nli_model.eval()

    def clean_and_split(self, text: str) -> List[str]:
        if not text: return []
        text = re.sub(r'==\s*[^=]+\s*==', '', text)
        # Giữ nguyên bản văn bản gốc Wikipedia để tránh lỗi ngữ pháp do Regex viết lại
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw_sentences if len(s.strip()) > 15 and not s.strip().startswith('|')]

    def select(self, claim: str, evidences: List[Evidence], top_n=2) -> List[Evidence]:
        candidates = []
        metadata = []

        for e in evidences:
            sentences_in_doc = self.clean_and_split(e.text)
            for idx, s in enumerate(sentences_in_doc):
                candidates.append(s)
                metadata.append({"source": e.source, "sentence_id": idx})

        if not candidates:
            return []

        # --- TẦNG 1: LỌC THÔ TOP 30 BẰNG MINI LM ---
        pairs = [(claim, sent) for sent in candidates]
        relevance_scores = self.cross_encoder.predict(pairs, activation_fct=torch.nn.Sigmoid())

        ranked_by_relevance = sorted(zip(candidates, metadata, relevance_scores), key=lambda x: x[2], reverse=True)
        top_30_candidates = ranked_by_relevance[:30]

        # --- TẦNG 2: NLI RERANK THEO CÔNG THỨC MENTOR ---
        final_sentences = [item[0] for item in top_30_candidates]
        final_metadata = [item[1] for item in top_30_candidates]
        final_rel_scores = [item[2] for item in top_30_candidates]

        inputs = self.nli_tokenizer(
            final_sentences,
            [claim] * len(final_sentences),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        hybrid_ranked = []
        for i in range(len(final_sentences)):
            rel_score = float(final_rel_scores[i])
            p_support = float(probs[i][0])
            p_refute = float(probs[i][1])
            
            # Công thức điểm hỗn hợp chuẩn bài của mentor
            max_logic = max(p_support, p_refute)
            hybrid_score = 0.2 * rel_score + 0.8 * max_logic

            hybrid_ranked.append((final_sentences[i], final_metadata[i], hybrid_score))

        hybrid_ranked = sorted(hybrid_ranked, key=lambda x: x[2], reverse=True)

        # Trả về Top 2 câu đơn tinh túy nhất
        final_evidences = []
        for sent, meta, score in hybrid_ranked[:top_n]:
            final_evidences.append(Evidence(text=sent, source=meta["source"], sentence_id=meta["sentence_id"], score=float(score)))
        return final_evidences

sentence_selector = SentenceSelector()
ranking_service = RankingService()