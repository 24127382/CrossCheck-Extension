# Delete later, this is just a placeholder for the retrieve service implementation
# import requests
# import spacy
# from fastapi import HTTPException
# from ..schemas.response import FactCheckEvidenceResponse

# # Load spacy model for NER
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     print("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
#     nlp = None

# async def retrieve(claim: str):
#     # Extract key entity (topic) using Named Entity Recognition
#     if nlp:
#         doc = nlp(claim)

#         priority = [
#             "EVENT",
#             "PRODUCT",
#             "ORG",
#             "PERSON",
#             "GPE"
#         ]

#         entities = sorted(
#             doc.ents,
#             key=lambda e:
#                 priority.index(e.label_)
#                 if e.label_ in priority
#                 else 999
#         )

#         topic = (
#             entities[0].text
#             if entities
#             else claim.split()[0]
#         )
#     else:
#         # Fallback to first word if spacy not available
#         topic = claim.split()[0]
    
#     url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"

#     headers = {
#         "User-Agent": "CrossCheckFactChecker/1.0 (your-email@example.com)"
#     }

#     try:
#         print(f"[Retrieve] Fetching Wikipedia for topic: {topic}")
#         print(f"[Retrieve] URL: {url}")
        
#         response = requests.get(url, headers=headers, timeout=10)

#         # Kiểm tra lỗi HTTP nếu có
#         response.raise_for_status()

#         data = response.json()
        
#         if "extract" not in data:
#             print(f"[Retrieve] ⚠️ No extract field in response: {list(data.keys())}")
#             raise Exception("Wikipedia API returned no extract field")

#         print(f"[Retrieve] ✅ Retrieved {len(data['extract'])} characters from Wikipedia")
        
#         return [
#             FactCheckEvidenceResponse(
#                 source="Wikipedia", content=data["extract"]
#             )
#         ]

#     except requests.exceptions.Timeout:
#         print("[Retrieve] ❌ Wikipedia API timeout")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Wikipedia API timeout while fetching topic: {topic}",
#         )
#     except requests.exceptions.HTTPError as http_err:
#         print(f"[Retrieve] ❌ HTTP Error: {http_err}")
#         print(f"[Retrieve] Response status: {response.status_code}")
#         print(f"[Retrieve] Response text: {response.text[:200]}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Wikipedia API error ({response.status_code}): {response.text[:100]}",
#         )
#     except Exception as err:
#         print(f"[Retrieve] ❌ Unexpected error: {str(err)}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(
#             status_code=500, detail=f"Error retrieving evidence: {str(err)}"
#         )