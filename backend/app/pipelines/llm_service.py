# Delete later, this is just a placeholder for the LLM service implementation
# import json
# from typing import List
# from pydantic import BaseModel, Field
# from google import genai
# from datetime import datetime
# from ..core.config import settings

# # Import đúng dataclass của bạn vào đây
# # Giả sử file chứa dataclass tên là schemas.evidence nha, bạn tự chỉnh lại cho đúng path
# from ..schemas.evidence import Evidence

# # Global counter for Gemini API calls
# gemini_call_count = 0 

# client = genai.Client(api_key=settings.AI_STUDIO_KEY)

# # Định nghĩa khuôn mẫu JSON để ép Gemini trả về đúng format
# class FactCheckReview(BaseModel):
#     verdict: str = Field(description="Kết luận cuối cùng. Bắt buộc thuộc một trong các giá trị: SUPPORTS, CONTRADICTS, NOT_ENOUGH_INFO")
#     confidence: float = Field(description="Độ tự tin của kết luận dựa trên chứng cứ, nhận giá trị số thực từ 0.0 đến 1.0")
#     summary: str = Field(description="Đoạn văn tóm tắt lập luận bằng tiếng Việt, giải thích rõ tại sao lại đưa ra kết luận đó dựa trên các chứng cứ")

# async def summarize(claim: str, evidences: List[Evidence]) -> dict:
#     """
#     Analyze claim against evidences using Gemini API
#     """
#     global gemini_call_count
#     gemini_call_count += 1
    
#     try:
#         print("[Gemini] Building evidence text...")
        
#         # Iterate through Evidence dataclass list, get source and text attributes
#         evidence_text = "\n".join([
#             f"- [{e.source}]: {e.text[:500]}"  # Limit to 500 chars per evidence to avoid token overflow
#             for e in evidences[:3]  # Limit to top 3 evidences to save tokens
#         ])
        
#         print(f"[Gemini] Evidence text length: {len(evidence_text)} characters")
        
#         prompt = f"""
# Analyze this claim against the provided evidence:

# Claim: "{claim}"

# Evidence:
# {evidence_text}

# Provide your analysis in JSON format with:
# - verdict: SUPPORTS, CONTRADICTS, or NOT_ENOUGH_INFO
# - confidence: 0.0 to 1.0
# - summary: Brief summary in English
# """
        
#         print(f"[Gemini] Prompt length: {len(prompt)} characters")
#         print("[Gemini] Sending request to Gemini API...")
#         print(f"[Gemini] API Key available: {bool(settings.AI_STUDIO_KEY)}")
#         print(f"[Gemini] Claim: {claim[:80]}...")
#         print(f"[Gemini] Evidence count: {len(evidences)}")
#         print(f"[Gemini] >>> CALL #{gemini_call_count} - {datetime.now().isoformat()} - CALLING GEMINI API NOW <<<")
        
#         # Gọi Gemini và ép trả về JSON cấu trúc
#         response = client.models.generate_content(
#             model="gemini-2.0-flash",
#             contents=prompt,
#             config={
#                 "response_mime_type": "application/json",
#                 "response_schema": FactCheckReview,
#             }
#         )
        
#         print(f"[Gemini] ✅ Response received (status: OK)")
#         print(f"[Gemini] Response text: {response.text[:200]}")
        
#         # Parse chuỗi JSON text thành Dictionary trong Python
#         result = json.loads(response.text)
#         print(f"[Gemini] ✅ Parsed JSON: {result}")
#         print(f"[Gemini] <<< CALL #{gemini_call_count} COMPLETED >>>")
#         return result
        
#     except json.JSONDecodeError as e:
#         print(f"[Gemini] ❌ JSON decode error: {str(e)}")
#         print(f"[Gemini] Raw response: {response.text[:500]}")
#         raise Exception(f"Failed to parse Gemini response: {str(e)}")
#     except Exception as e:
#         print(f"[Gemini] ❌ API Error: {str(e)}")
#         import traceback
#         traceback.print_exc()
        
#         # Check for specific error types
#         error_str = str(e).lower()
#         if "quota" in error_str or "rate" in error_str:
#             raise Exception("Gemini API quota exceeded. Try again later.")
#         elif "authentication" in error_str or "api key" in error_str or "unauthorized" in error_str:
#             raise Exception("Gemini API key is invalid or not configured.")
#         elif "token" in error_str:
#             raise Exception("Token limit exceeded. Claim or evidence too long.")
#         else:
#             raise Exception(f"Gemini API error: {str(e)}")