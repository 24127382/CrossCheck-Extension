import requests
from fastapi import HTTPException

class WikipediaClient:
    """Client for interacting with Wikipedia API"""
    def search(self, query: str):
        HEADERS = {"User-Agent": "CrossCheckExtension/1.0 (https://github.com/24127382/crosscheck; hoang.trn0811@gmail.com)"}
        url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json"}
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        return response.json()

    def extract(self, title: str):
        HEADERS = {"User-Agent": "CrossCheckExtension/1.0 (https://github.com/24127382/crosscheck; hoang.trn0811@gmail.com)"}
        url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "prop": "extracts", "explaintext": True, "titles": title, "format": "json"}
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        return response.json()

wiki_client = WikipediaClient()