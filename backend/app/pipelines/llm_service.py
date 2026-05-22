import json
from typing import List
from pydantic import BaseModel, Field
from google import genai
from app.core.config import settings

# Import đúng dataclass của bạn vào đây
# Giả sử file chứa dataclass tên là schemas.evidence nha, bạn tự chỉnh lại cho đúng path
from app.schemas.evidence import Evidence 

client = genai.Client(api_key=settings.AI_STUDIO_KEY)

# Định nghĩa khuôn mẫu JSON để ép Gemini trả về đúng format
class FactCheckReview(BaseModel):
    verdict: str = Field(description="Kết luận cuối cùng. Bắt buộc thuộc một trong các giá trị: SUPPORTS, CONTRADICTS, NOT_ENOUGH_INFO")
    confidence: float = Field(description="Độ tự tin của kết luận dựa trên chứng cứ, nhận giá trị số thực từ 0.0 đến 1.0")
    summary: str = Field(description="Đoạn văn tóm tắt lập luận bằng tiếng Việt, giải thích rõ tại sao lại đưa ra kết luận đó dựa trên các chứng cứ")

async def summarize(claim: str, evidences: List[Evidence]) -> dict:
    """
    Nhận vào claim và list Evidence (đã qua lọc/rank), 
    nhờ Gemini phân tích toán bộ để trả về JSON phân tích cuối cùng.
    """
    
    # Duyệt qua danh sách dataclass Evidence, lấy nguồn (.source) và nội dung (.text)
    #  SỬA THÀNH: Sử dụng .content thay vì .text và bỏ .stance đi
    evidence_text = "\n".join([
        f"- Nguồn [{e.source}]: {e.content}" 
        for e in evidences
    ])
    
    prompt = f"""
    Bạn là một chuyên gia kiểm định thông tin cấp cao. Hãy đối chiếu Tuyên bố (Claim) với danh sách các Chứng cứ (Evidence) được cung cấp dưới đây để đưa ra kết luận cuối cùng.
    
    Tuyên bố cần kiểm định: "{claim}"
    
    Danh sách chứng cứ tìm được:
    {evidence_text}
    """
    
    # Gọi Gemini và ép trả về JSON cấu trúc
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": FactCheckReview,
        }
    )
    
    # Parse chuỗi JSON text thành Dictionary trong Python
    return json.loads(response.text)