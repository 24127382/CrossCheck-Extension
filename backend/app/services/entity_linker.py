'''A simple entity linker that uses Wikipedia search results and a heuristic scoring mechanism.'''
import re
from typing import List, Dict
from difflib import SequenceMatcher
from .wikipedia_client import wiki_client, WikipediaClient  # Import một chiều từ hạ tầng

class EntityLinker:
    """Lightweight Wikipedia Entity Linking (no LLM)"""
    def __init__(self, client: WikipediaClient = None):
        self.client = client if client else wiki_client

    def extract_mentions(self, doc) -> List[str]:
        return list(set([ent.text for ent in doc.ents]))

    def similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def link(self, mention: str, top_k: int = 5) -> str:
        if not mention or len(mention.strip()) <= 2:
            return None
        results = self.client.search(mention)
        candidates = results.get("query", {}).get("search", [])

        scored = []
        for c in candidates[:top_k]:
            title = c["title"]
            snippet = c.get("snippet", "")
            normalized_size = min(c.get("size", 0) / 150000, 1.0)

            score = (
                0.6 * self.similarity(mention, title)
                + 0.2 * min(len(snippet) / 200, 1.0)
                + 0.2 * normalized_size
            )
            scored.append((title, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None

# Khởi tạo object toàn cục
entity_linker = EntityLinker(client=wiki_client)
