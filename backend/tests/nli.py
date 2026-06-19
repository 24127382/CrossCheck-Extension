import torch

from app.services.entailment_service import entailment_service
from datasets import load_dataset
import asyncio
async def test_entailment_service():
    claim = (
    "The Indian National Congress "
    "was founded in Bombay, India."
)

    evidence = (
        "In 1885, the Indian National Congress "
        "was founded at Gokuldas Tejpal "
        "Sanskrit College in Bombay."
    )

    for order in ["evidence_first", "claim_first"]:

        if order == "evidence_first":
            inputs = entailment_service.tokenizer(
                evidence,
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
        else:
            inputs = entailment_service.tokenizer(
                claim,
                evidence,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        inputs = {
            k: v.to(entailment_service.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            logits = entailment_service.model(**inputs).logits
            probs =  torch.softmax(logits, dim=1)

        print(order)
        print(probs.cpu().numpy())
    
if __name__ == "__main__":
    asyncio.run(test_entailment_service())