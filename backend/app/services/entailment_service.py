import torch
import numpy as np
from typing import Dict, List
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


class EntailmentService:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        repo_id = "Akiya-Vyre/deberta-v3-fever"

        print(
            "[Entailment] Loading model..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                repo_id
            )
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
            2: "NOT_ENOUGH_INFO"
        }

        print(
            "[Entailment] Ready."
        )

    def _prepare_inputs(
    self,
    claim: str,
    evidences: List[str]
    ):

        inputs = self.tokenizer(
            evidences,                 # text1 = evidence
            [claim] * len(evidences), # text2 = claim
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        return {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

    def _default_nei(self):

        return {
            "verdict": "NOT_ENOUGH_INFO",
            "confidence": 1.0,
            "scores": {
                "supports": 0.0,
                "refutes": 0.0,
                "nei": 1.0
            }
        }

    async def predict_verdict(
    self,
    claim: str,
    evidences: List[str]
    ) -> Dict:

        if not evidences:
            return self._default_nei()

        print(type(evidences))

        # for i, e in enumerate(evidences):
        #     print(
        #         i,
        #         type(e),
        #         repr(e)[:100]
        #     )
        
        inputs = self._prepare_inputs(
            claim,
            evidences
        )

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

            probs = torch.softmax(
                outputs.logits,
                dim=1
            ).cpu().numpy()

        # ==========================================
        # Aggregation
        # ==========================================

        support_scores = probs[:, 0]
        refute_scores = probs[:, 1]
        nei_scores = probs[:, 2]

        max_support = float(
            np.max(support_scores)
        )

        max_refute = float(
            np.max(refute_scores)
        )

        mean_nei = float(
            np.mean(nei_scores)
        )

        # ==================================================
        # Evidence Strength Decision
        # ==================================================

        EVIDENCE_TH = 0.75
        MARGIN_TH = 0.10

        evidence_strength = max(
            max_support,
            max_refute
        )

        # Không có bằng chứng đủ mạnh
        if evidence_strength < EVIDENCE_TH:

            pred = 2
            confidence = 1.0 - evidence_strength

        else:

            # SUPPORT thắng rõ
            if max_support > max_refute + MARGIN_TH:

                pred = 0
                confidence = max_support

            # REFUTE thắng rõ
            elif max_refute > max_support + MARGIN_TH:

                pred = 1
                confidence = max_refute

            # Hai bên quá sát nhau
            else:

                pred = 2
                confidence = max(
                    mean_nei,
                    1.0 - abs(max_support - max_refute)
                )

        verdict_map = {
            0: "SUPPORTS",
            1: "REFUTES",
            2: "NOT_ENOUGH_INFO"
        }

        return {
            "verdict": verdict_map[pred],

            "confidence": float(confidence),

            "scores": {
                "supports": max_support,
                "refutes": max_refute,
                "nei": mean_nei
            },

            "debug": {
                "num_evidences": len(evidences),

                "evidence_strength": evidence_strength,

                "support_minus_refute":
                    max_support - max_refute,

                "raw_probs": [
                    {
                        "supports": float(p[0]),
                        "refutes": float(p[1]),
                        "nei": float(p[2])
                    }
                    for p in probs
                ]
            }
        }

entailment_service = (
    EntailmentService()
)