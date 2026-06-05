"""Retrieval service - handles evidence retrieval from Wikipedia"""
from typing import List
import requests
import spacy
from fastapi import HTTPException
from ..schemas.evidence import Evidence
from ..schemas.response import FactCheckEvidenceResponse

# Load spacy model for NER
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Spacy model not found. Please run: python -m spacy download en_core_web_lg")
    nlp = None

class RetrievalService:
    """Service for retrieving relevant evidence for claims from Wikipedia"""
    
    def __init__(self, nlp):
        # Initialize retrieval model (e.g., BM25, dense retriever)
        self.nlp = nlp
    
    async def retrieve_evidence(self, claim: str, top_k: int = 10) -> List[Evidence]:
        """
        Retrieve relevant evidence documents for a claim from Wikipedia
        
        Args:
            claim: The claim to find evidence for
            top_k: Number of top evidence items to return
            
        Returns:
            List of Evidence objects
        """
        headers = {
            "User-Agent": "CrossCheckFactChecker/1.0 (your-email@example.com)"
        }
        
        priority = {"EVENT", "PRODUCT", "ORG", "PERSON", "GPE", "LOC", "FAC", "WORK_OF_ART"}
        # ======= log  ============
        print("NLP:", self.nlp)
        
        
        # Extract key entities (topics) using Named Entity Recognition
        topics = []
        if self.nlp:
            doc = self.nlp(claim)
            seen = set()
            for ent in doc.ents:
                # log to see
                print(ent.text, ent.label_)
                if ent.label_ in priority and ent.text not in seen:
                    topics.append(ent.text)
                    seen.add(ent.text)
        
        for chunk in doc.noun_chunks:
                # Loại bỏ các từ bắt đầu bằng đại từ hoặc từ nối chung chung nếu cần
                if chunk.root.is_stop or len(chunk.text.strip()) <= 3:
                    continue
                if chunk.root.pos_ not in ["NOUN", "PROPN"]: # Chỉ lấy Danh từ/Danh từ riêng
                    continue
                    
                chunk_text = chunk.text.strip()
                if chunk_text.lower() not in seen:
                    topics.append(chunk_text)
                    seen.add(chunk_text.lower())
        
        topics = topics[:top_k]
        results = []
        
        for topic in topics:
            try:
                # --- THAY ĐỔI QUAN TRỌNG: DÙNG WIKIPEDIA SEARCH API TRƯỚC ---
                # Vì "about a third" hay "Everglades" viết thường có thể làm sập API summary trực tiếp.
                search_url = "https://en.wikipedia.org/w/api.php"
                search_params = {
                    "action": "query",
                    "prop": "extracts",
                    "exintro": True,       # Lấy toàn bộ phần Intro (thường là 3-4 đoạn đầy đủ)
                    "explaintext": True,   # Lấy text thuần, bỏ qua mã HTML
                    "titles": actual_title,
                    "format": "json"
                }
                
                search_res = requests.get(search_url, params=search_params, headers=headers, timeout=5)
                if search_res.status_code != 200:
                    continue
                    
                search_data = search_res.json()
                search_results = search_data.get("query", {}).get("search", [])
                
                if not search_results:
                    print(f"[Retrieve] ⚠️ Không tìm thấy bài viết nào trên Wiki cho từ khóa: {topic}")
                    continue
                
                # Lấy ra Title chuẩn nhất được Wiki gợi ý
                actual_title = search_results[0]["title"]
                formatted_topic = actual_title.replace(" ", "_")
                
                # Bây giờ mới gọi API Summary bằng Title chuẩn
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_topic}"
                response = requests.get(url, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if "extract" in data:
                        evidence = Evidence(
                            source=f"Wikipedia: {actual_title}",
                            stance="NEUTRAL",
                            score=1.0,
                            text=data["extract"]
                        )
                        results.append(evidence)
                        print(f"[Retrieve] ✅ Thành công với từ khóa chuẩn: {actual_title} (gốc: {topic})")
            
            except Exception as e:
                print(f"[Retrieve] ❌ Error: {str(e)}")
                continue
                
        return results
            

retrieval_service = RetrievalService(nlp)
