"""Entailment service - handles textual entailment using roberta"""
import torch
import numpy as np
from typing import Dict, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download

from ..schemas.evidence import Evidence, EvidenceScore

class EntailmentService:
    """Service for NLI (Natural Language Inference) / Textual Entailment using RoBERTa-MNLI"""
    
    def __init__(self):
        # Initialize entailment model (RoBERTa fine-tuned on MNLI)
        print("[Entailment] Loading DeBERTa FEVER model...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        repo_id = "Akiya-Vyre/deberta-v3-fever"

        self.tokenizer = AutoTokenizer.from_pretrained(
            repo_id
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(repo_id)
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            0: "SUPPORTS",
            1: "REFUTES",
            2: "NEI"
        }

        print(
            "[Entailment] "
            "DeBERTa model loaded"
        )
        
    
    async def predict_verdict(
        self,
        claim: str,
        pseudo_outline: str
    ):
        inputs = self.tokenizer(
            claim,
            pseudo_outline,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

            probs = torch.softmax(
                outputs.logits,
                dim=1
            )[0].cpu().numpy()

        supports = float(probs[0])
        refutes = float(probs[1])
        nei = float(probs[2])

        pred = int(np.argmax(probs))

        verdict_map = {
            0: "SUPPORTS",
            1: "REFUTES",
            2: "NOT_ENOUGH_INFO"
        }

        return {
            "verdict": verdict_map[pred],
            "confidence": float(probs[pred]),
            "scores": {
                "supports": supports,
                "refutes": refutes,
                "nei": nei
            }
        }

entailment_service = EntailmentService()