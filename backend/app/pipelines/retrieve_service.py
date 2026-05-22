import requests
from fastapi import HTTPException
from app.schemas.response import FactCheckEvidenceResponse

async def retrieve(claim: str):
    topic = "Earth"
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"

    # ➕ THÊM ĐOẠN HEADERS NÀY VÀO
    headers = {
        # Đặt tên ứng dụng của bạn để Wikipedia biết bạn là ai và không chặn nữa
        "User-Agent": "CrossCheckFactChecker/1.0 (your-email@example.com)"
    }

    try:
        # Truyền thêm headers=headers vào đây
        response = requests.get(url, headers=headers)

        # Kiểm tra lỗi HTTP nếu có
        response.raise_for_status()

        data = response.json()

        # Trả về kết quả (Nhớ kiểm tra class Response của bạn tên gì nha)
        # Ví dụ ở đây tui giả định bạn dùng FactCheckEvidenceResponse
        from app.schemas.response import FactCheckEvidenceResponse

        return [
            FactCheckEvidenceResponse(
                source="Wikipedia", content=data["extract"]
            )
        ]

    except requests.exceptions.HTTPError as http_err:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi gọi API Wikipedia: {http_err}. Nội dung: {response.text[:100]}",
        )
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Lỗi hệ thống không xác định: {err}"
        )