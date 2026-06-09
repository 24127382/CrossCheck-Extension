"""Entailment service - handles textual entailment using roberta"""
import torch
from typing import Dict, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download

from ..schemas.evidence import Evidence, EvidenceScore

class EntailmentService:
    """Service for NLI (Natural Language Inference) / Textual Entailment using RoBERTa-MNLI"""
    
    def __init__(self):
        # Initialize entailment model (RoBERTa fine-tuned on MNLI)
        print("[Entailment] Loading RoBERTa model...")
        self.checkpoint_path = hf_hub_download(repo_id="Akiya-Vyre/mocheg-roberta", filename="best_model.pt")
        self.model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        print("[Entailment] ✅ RoBERTa model loaded")
        
    async def compute_entailment(self, claim: str, evidence_list: List[Evidence]) -> List[EvidenceScore]:
        """
        Compute entailment scores between claim and evidence
        
        Args:
            claim: The claim being fact-checked
            evidence_list: List of Evidence objects to evaluate against claim
            
        Returns:
            List of EvidenceScore objects with entailment scores
        """
        print(f"[Entailment] Computing entailment for claim vs {len(evidence_list)} evidences...")
        
        results = []
        for evidence in evidence_list:
            try:
                # Tokenize claim vs evidence text
                inputs = self.tokenizer(
                    claim,
                    evidence.text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512 # mocheg context window limit
                )
                
                # Run model inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # probabilities: [contradiction, neutral, entailment]
                entailment_score = float(probabilities[2])  # entailment probability
                
                print(f"[Entailment] {evidence.source}: entail={entailment_score:.3f}")
                
                # Create EvidenceScore with entailment information
                evidence_score = EvidenceScore(
                    evidence=evidence,
                    relevance_score=evidence.score,  # From retrieval service
                    sentence_score=getattr(evidence, 'score', 0.0), # Giữ lại trạng thái điểm cũ nếu có
                    entailment_score=entailment_score,
                    # FIX: KHÔNG ghi đè final_score tại đây để giữ tầng gộp điểm (aggregation) chạy chuẩn xác
                    final_score=evidence.score  
                )
                results.append(evidence_score)
                
            except Exception as e:
                print(f"[Entailment] ❌ Error processing '{evidence.source}': {str(e)}")
                continue
        
        print(f"[Entailment] ✅ Computed entailment for {len(results)}/{len(evidence_list)} evidences")
        return results
    
    async def predict_verdict(self, claim: str, pseudo_outline: str) -> dict:

        print(f"[Entailment] Predicting verdict for claim: {claim[:80]}...")

        try:
            combined_input = f"{claim} </s> {pseudo_outline}"

            inputs = self.tokenizer(
                combined_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )

            logits = outputs.logits

            probabilities = torch.softmax(
                logits,
                dim=1
            ).cpu().numpy()[0]

            entailment_score = float(probabilities[0])
            contradiction_score = float(probabilities[1])
            neutral_score = float(probabilities[2])

            print(
                f"[Entailment] Scores -> "
                f"E={entailment_score:.3f} "
                f"C={contradiction_score:.3f} "
                f"N={neutral_score:.3f}"
            )

            if entailment_score > contradiction_score and entailment_score > neutral_score:
                verdict = "SUPPORTS"
                confidence = entailment_score

            elif contradiction_score > entailment_score and contradiction_score > neutral_score:
                verdict = "CONTRADICTS"
                confidence = contradiction_score

            else:
                verdict = "NOT_ENOUGH_INFO"
                confidence = neutral_score

            print(
                f"[Entailment" f"] ✅ Predicted verdict: "
                f"{verdict} ({confidence:.3f})"
            )

            return {
                "verdict": verdict,
                "confidence": confidence,
                "scores": {
                    "entailment": entailment_score,
                    "contradiction": contradiction_score,
                    "neutral": neutral_score
                }
            }

        except Exception as e:
            print(f"[Entailment] ❌ Error: {str(e)}")

            return {
                "verdict": "NOT_ENOUGH_INFO",
                "confidence": 0.0,
                "scores": {}
            }

entailment_service = EntailmentService()