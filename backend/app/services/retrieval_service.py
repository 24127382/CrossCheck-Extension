"""Retrieval service - handles evidence retrieval from Wikipedia"""
from typing import List
import requests
import spacy
from fastapi import HTTPException
from ..schemas.evidence import Evidence
from ..schemas.response import FactCheckEvidenceResponse
from ..schemas import response

# Load spacy model for NER
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Spacy model not found. Please run: python -m spacy download en_core_web_lg")
    nlp = None

class WikipediaClient:
    """Client for interacting with Wikipedia API"""
    
    def search(self, query: str):
        HEADERS = {
            "User-Agent": (
                "CrossCheckExtension/1.0 "
                "(https://github.com/yourname/crosscheck; your@email.com)"
            )
        }
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        
        print("\n========== DEBUG ==========")
        print("QUERY:", query)
        print("STATUS:", response.status_code)
        print("CONTENT-TYPE:", response.headers.get("content-type"))
        print("BODY:")
        print(response.text[:1000])
        print("===========================\n")
        return response.json()

    def extract(self, title: str):
        HEADERS = {
            "User-Agent": (
                "CrossCheckExtension/1.0 "
                "(https://github.com/yourname/crosscheck; your@email.com)"
            )
        }
        url = "https://en.wikipedia.org/w/api.php"
        
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "titles": title,
            "format": "json"
        }
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        print("\n========== EXTRACT DEBUG ==========")
        print("TITLE:", title)
        print("STATUS:", response.status_code)
        print("CONTENT-TYPE:", response.headers.get("content-type"))
        print("BODY:")
        print(response.text[:1000])
        print("===========================\n")
        return response.json()

class QueryBuilder:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def extract_topics(self, text: str):
        doc = self.nlp(text) if self.nlp else None

        if not doc:
            return [text]

        entities = [ent.text for ent in doc.ents]
        nouns = [c.text for c in doc.noun_chunks]

        seen = set()
        result = []

        for t in entities + nouns:
            if t.lower() not in seen:
                result.append(t)
                seen.add(t.lower())

        return result[:5]

class RetrievalService:
    def __init__(self):
        self.client = WikipediaClient()

    def retrieve(self, topics):
        evidences = []

        for topic in topics:
            search = self.client.search(topic)
            results = search.get("query", {}).get("search", [])

            if not results:
                continue

            best = max(
                results,
                key=lambda x: x.get("size", 0) 
            )
            
            title = best["title"]
            extract = self.client.extract(title)

            pages = extract.get("query", {}).get("pages", {})
            page = list(pages.values())[0]
            text = page.get("extract", "")

            if text:
                evidences.append(
                    Evidence(text=text, source=title)
                )

        return evidences
            

retrieval_service = RetrievalService()
query_builder = QueryBuilder()
wiki_client = WikipediaClient()