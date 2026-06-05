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

entailment_service = EntailmentService()
