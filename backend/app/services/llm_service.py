import json
from typing import List
from pydantic import BaseModel, Field
# Thay thế google-genai bằng huggingface_hub
from huggingface_hub import InferenceClient 
from datetime import datetime
from ..core.config import settings
from ..schemas.evidence import Evidence

# Global counter for Hugging Face API calls
huggingface_call_count = 0 

# Khởi tạo client của Hugging Face 
client = InferenceClient(token=settings.HF_TOKEN)

# Định nghĩa khuôn mẫu JSON bằng Pydantic để lấy JSON schema làm hướng dẫn cấu trúc cho model
class FactCheckReview(BaseModel):
    verdict: str = Field(description="Kết luận cuối cùng. Bắt buộc thuộc một trong các giá trị: SUPPORTS, CONTRADICTS, NOT_ENOUGH_INFO")
    confidence: float = Field(description="Độ tự tin của kết luận dựa trên chứng cứ, nhận giá trị số thực từ 0.0 đến 1.0")
    summary: str = Field(description="Đoạn văn tóm tắt lập luận bằng tiếng Việt, giải thích rõ tại sao lại đưa ra kết luận đó dựa trên các chứng cứ")

class LLMService:
    """Service for LLM-based operations using Hugging Face Free Inference API"""
    
    def __init__(self):
        self.client = client
        self.call_count = 0
        # Đổi sang Qwen 2.5 để chạy ngay lập tức không cần đăng ký gated model, hiểu tiếng Việt rất tốt
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    async def summarize(self, claim: str, evidences: List[Evidence]) -> dict:
        """
        Analyze claim against evidences using Hugging Face API
        """
        global huggingface_call_count
        huggingface_call_count += 1
        self.call_count += 1
        
        try:
            print("[HuggingFace] Building evidence text...")
            
            evidence_text = "\n".join([
                f"- [{e.source}]: {e.text[:500]}"  
                for e in evidences[:3]  
            ])
            
            print(f"[HuggingFace] Evidence text length: {len(evidence_text)} characters")
            
            # Viết prompt rõ ràng và yêu cầu trả về theo schema cụ thể
            prompt = f"""
                You are an expert fact-checker. Analyze this claim against the provided evidence.
                You must respond ONLY with a JSON object that strictly adheres to this JSON Schema:
                {json.dumps(FactCheckReview.model_json_schema(), ensure_ascii=False)}

                Claim: "{claim}"

                Evidence:
                {evidence_text if evidence_text else "No evidence provided. Analyze based on your general knowledge."}
                """
                            
            print(f"[HuggingFace] Prompt length: {len(prompt)} characters")
            print(f"[HuggingFace] Sending request to HF API ({self.model_name})...")
            print(f"[HuggingFace] >>> CALL #{huggingface_call_count} - {datetime.now().isoformat()} - CALLING API NOW <<<")
            
            # Gọi Hugging Face Inference API với cấu trúc sửa lỗi format JSON
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                # Sửa đổi thành cấu trúc json_object chuẩn mà TogetherAI/Novita yêu cầu
                response_format={
                    "type": "json_object",
                    "schema": FactCheckReview.model_json_schema()
                }
            )
            
            # Lấy nội dung text từ response chat_completion
            response_text = response.choices[0].message.content
            
            print(f"[HuggingFace] ✅ Response received (status: OK)")
            print(f"[HuggingFace] Response text: {response_text[:200]}")
            
            # Parse chuỗi JSON text thành Dictionary trong Python
            result = json.loads(response_text)
            print(f"[HuggingFace] ✅ Parsed JSON: {result}")
            print(f"[HuggingFace] <<< CALL #{huggingface_call_count} COMPLETED >>>")
            return result
            
        except json.JSONDecodeError as e:
            print(f"[HuggingFace] ❌ JSON decode error: {str(e)}")
            print(f"[HuggingFace] Raw response: {response_text[:500] if 'response_text' in locals() else 'None'}")
            raise Exception(f"Failed to parse HuggingFace response: {str(e)}")
        except Exception as e:
            print(f"[HuggingFace] ❌ API Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            error_str = str(e).lower()
            if "rate limit" in error_str or "too many requests" in error_str:
                raise Exception("Hugging Face API free quota exceeded. Try again later.")
            elif "token" in error_str or "authorization" in error_str:
                raise Exception("Hugging Face Token is invalid or missing.")
            else:
                raise Exception(f"Hugging Face API error: {str(e)}")
    
    async def generate_verdict(self, claim: str, evidence_text: str) -> dict:
        return await self.summarize(claim, [])

llm_service = LLMService()