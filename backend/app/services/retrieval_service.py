"""Retrieval service - handles evidence retrieval from Wikipedia"""
from typing import List
import requests
import spacy
from fastapi import HTTPException
from ..schemas.evidence import Evidence
from ..schemas.response import FactCheckEvidenceResponse

# Load spacy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

class RetrievalService:
    """Service for retrieving relevant evidence for claims from Wikipedia"""
    
    def __init__(self):
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
        
        priority = [
            "EVENT",
            "PRODUCT",
            "ORG",
            "PERSON",
            "GPE"
        ]
        
        # Extract key entities (topics) using Named Entity Recognition
        topics = []
        if self.nlp:
            doc = self.nlp(claim)
            seen = set()
            for ent in doc.ents:
                if ent.label_ in priority and ent.text not in seen:
                    topics.append(ent.text)
                    seen.add(ent.text)
        
        # Fallback to first word if no entities found
        if not topics:
            topics = [claim.split()[0]]
        
        topics = topics[:top_k]
        results = []
        
        for topic in topics:
            try:
                # Normalize topic for Wikipedia API query
                formatted_topic = topic.replace(" ", "_")
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_topic}"
                
                print(f"[Retrieve] Fetching Wikipedia for topic: {topic}")
                response = requests.get(url, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if "extract" in data:
                        evidence = Evidence(
                            source=f"Wikipedia: {topic}",
                            stance="NEUTRAL",
                            score=1.0,
                            text=data["extract"]
                        )
                        results.append(evidence)
                        print(f"[Retrieve] ✅ Retrieved evidence for topic: {topic}")
                else:
                    print(f"[Retrieve] ⚠️ Failed to retrieve evidence for topic: {topic} (status: {response.status_code})")
            except Exception as e:
                print(f"[Retrieve] ❌ Error retrieving evidence for topic '{topic}': {str(e)}")
                continue
        
        print(f"[Retrieve] Total evidences retrieved: {len(results)}")
        return results
            

retrieval_service = RetrievalService()
