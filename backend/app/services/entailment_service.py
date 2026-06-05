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
                    max_length=512
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
                    entailment_score=entailment_score,
                    final_score=entailment_score  # Can be combined with relevance_score later
                )
                results.append(evidence_score)
                
            except Exception as e:
                print(f"[Entailment] ❌ Error processing '{evidence.source}': {str(e)}")
                continue
        
        print(f"[Entailment] ✅ Computed entailment for {len(results)}/{len(evidence_list)} evidences")
        return results
    
    async def predict_verdict(self, claim: str, evidence_list: List[Evidence]) -> dict:
        """
        Predict verdict and confidence based on entailment scores
        
        Args:
            claim: The claim being fact-checked
            evidence_list: List of Evidence objects
            
        Returns:
            Dict with verdict, confidence, and entailment scores
        """
        print(f"[Entailment] Predicting verdict for claim: {claim[:80]}...")
        
        if not evidence_list:
            return {
                "verdict": "NOT_ENOUGH_INFO",
                "confidence": 0.0,
                "entailment_scores": []
            }
        
        entailment_scores = []
        
        for evidence in evidence_list:
            try:
                inputs = self.tokenizer(
                    claim,
                    evidence.text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # probabilities: [contradiction, neutral, entailment]
                contradiction_score = float(probabilities[0])
                neutral_score = float(probabilities[1])
                entailment_score = float(probabilities[2])
                
                entailment_scores.append({
                    "source": evidence.source,
                    "contradiction": contradiction_score,
                    "neutral": neutral_score,
                    "entailment": entailment_score
                })
                
            except Exception as e:
                print(f"[Entailment] ❌ Error predicting for '{evidence.source}': {str(e)}")
                continue
        
        # Aggregate scores across all evidences
        if not entailment_scores:
            return {
                "verdict": "NOT_ENOUGH_INFO",
                "confidence": 0.0,
                "entailment_scores": []
            }
        
        avg_entailment = sum(s["entailment"] for s in entailment_scores) / len(entailment_scores)
        avg_contradiction = sum(s["contradiction"] for s in entailment_scores) / len(entailment_scores)
        avg_neutral = sum(s["neutral"] for s in entailment_scores) / len(entailment_scores)
        
        print(f"[Entailment] Aggregated scores: entail={avg_entailment:.3f}, contra={avg_contradiction:.3f}, neutral={avg_neutral:.3f}")
        
        # Determine verdict based on highest score
        if avg_entailment > avg_contradiction and avg_entailment > avg_neutral:
            verdict = "SUPPORTS"
            confidence = avg_entailment
        elif avg_contradiction > avg_entailment and avg_contradiction > avg_neutral:
            verdict = "CONTRADICTS"
            confidence = avg_contradiction
        else:
            verdict = "NOT_ENOUGH_INFO"
            confidence = avg_neutral
        
        print(f"[Entailment] ✅ Predicted verdict: {verdict} (confidence: {confidence:.3f})")
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "entailment_scores": entailment_scores
        }

entailment_service = EntailmentService()
