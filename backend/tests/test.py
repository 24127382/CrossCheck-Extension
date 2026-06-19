"""
Independent test script for NLI layer (DeBERTa) integrating EntailmentService.
Fully synchronized logic with the original Service file.
"""
import asyncio
import sys
import time
import urllib.parse
from datasets import load_dataset
import traceback
import random

sys.stdout.reconfigure(encoding='utf-8')

# Import services based on your architecture
from app.services.retrieval_service import query_builder, retrieval_service
from app.services.ranking_service import ranking_service, sentence_selector
from app.services.entailment_service import entailment_service

try:
    from app.services.retrieval_service import entity_linker
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    entity_linker = None


def clean_wiki_title(url_or_title: str) -> str:
    if not url_or_title: return ""
    if "wikipedia.org/wiki/" in url_or_title:
        url_or_title = url_or_title.split("wikipedia.org/wiki/")[-1]
    decoded_title = urllib.parse.unquote(url_or_title)
    cleaned = decoded_title.strip().replace(" ", "_").lower()
    cleaned = cleaned.replace("-lrb-", "(").replace("-rrb-", ")")
    return cleaned


def extract_gold_evidence_fixed(sample):
    """Accurately extract FEVER v1.0 4-level nested evidence structure"""
    gold_pages = set()
    gold_sentences_idx = []
    
    if "evidence" in sample and sample["evidence"]:
        for evidence_group in sample["evidence"]:
            for evidence_item in evidence_group:
                if isinstance(evidence_item, (list, tuple)) and len(evidence_item) >= 4:
                    page_title = evidence_item[2]
                    sent_idx = evidence_item[3]
                    if page_title:
                        gold_pages.add(page_title)
                    if sent_idx is not None and sent_idx != -1:
                        gold_sentences_idx.append(f"{page_title} [Sent {sent_idx}]")
                        
    page_str = ", ".join(list(gold_pages))
    sent_str = ", ".join(gold_sentences_idx)
    return page_str if page_str else "N/A", sent_str if sent_str else "N/A"


async def process_nli_sample(sample, current_idx, total_count):
    claim = sample["claim"]
    gold_label = sample["label"].upper().strip() # SUPPORTS or REFUTES

    # Get gold evidence from dataset
    gold_page, gold_sentence = extract_gold_evidence_fixed(sample)

    try:
        # 1. Extract topics & entities from Claim
        topics = query_builder.extract_topics(claim)
        entity_links = []
        if nlp and entity_linker:
            doc = nlp(claim)
            mentions = entity_linker.extract_mentions(doc)
            for m in mentions:
                linked = entity_linker.link(m)
                if linked: entity_links.append(linked)
        
        # 2. Retrieve documents from Wikipedia
        evidences = retrieval_service.retrieve(topics=topics, entity_links=entity_links)
        if not evidences:
            print(f" [{current_idx}/{total_count}] SKIP | Claim: {claim[:45]}... | Reason: Cannot retrieve pages")
            return None 

        retrieved_pages = ", ".join([clean_wiki_title(e.url if hasattr(e, 'url') else getattr(e, 'title', '')) for e in evidences])

        # 3. Rerank documents & select top sentences
        scored_documents = await ranking_service.rank_evidence(claim, evidences)
        scored_documents = sorted(scored_documents, key=lambda x: x.final_score, reverse=True)
        top_evidences = [e.evidence for e in scored_documents[:10]]
        
        sentence_evidences = sentence_selector.select(
            claim=claim, evidences=top_evidences, top_n=2
        )
        if not sentence_evidences:
            print(f" [{current_idx}/{total_count}] SKIP | Claim: {claim[:45]}... | Reason: Sentence Selector empty")
            return None

        selected_sentences_log = " | ".join([f"[{ev.page if hasattr(ev, 'page') else ''}]: {ev.text}" for ev in sentence_evidences])
        
        # Convert to list of sentence text
        pseudo_outline = [ev.text for ev in sentence_evidences]
        
        # 4. Predict verdict via batching service
        res_dict = await entailment_service.predict_verdict(claim=claim, evidences=pseudo_outline)
        
        # Extract results
        pred_label = res_dict.get("verdict", "NOT_ENOUGH_INFO").upper().strip()
        confidence = res_dict.get("confidence", 0.0)
        scores = res_dict.get("scores", {})
        debug_info = res_dict.get("debug", {})  
        
        is_correct = (pred_label == gold_label)
        status = "CORRECT" if is_correct else "WRONG"
        
        # --- DETAILED SYSTEM LOG (NO ACCENTS, NO ICONS) ---
        print("-" * 80)
        print(f"INDEX: {current_idx}/{total_count} | PIPELINE RESULT: {status}")
        print(f"CLAIM          : {claim}")
        print(f"GOLD PAGE      : {gold_page}")
        print(f"GOLD SENTENCE  : {gold_sentence}")
        print(f"GOLD LABEL     : {gold_label}")
        print(f"RETRIEVAL PAGE : {retrieved_pages[:150]}...")
        print(f"SENTENCE SELECT: {selected_sentences_log[:200]}...")
        print(f"PRED LABEL     : {pred_label} (Confidence: {confidence*100:.2f}%)")
        
        # Mentor Request 1: Print threshold decision scores
        print(
            f"SCORES -> "
            f"S:{scores.get('supports', 0.0):.3f} | "
            f"R:{scores.get('refutes', 0.0):.3f} | "
            f"N:{scores.get('nei', 0.0):.3f}"
        )
        
        # Mentor Request 2: Log raw probability of each sentence
        if "raw_probs" in debug_info and debug_info["raw_probs"]:
            print("\nEVIDENCE BREAKDOWN")
            for idx, (ev, p) in enumerate(
                zip(sentence_evidences, debug_info["raw_probs"]),
                1
            ):
                print(
                    f"[{idx}] "
                    f"S={p.get('supports', 0.0):.3f} "
                    f"R={p.get('refutes', 0.0):.3f} "
                    f"N={p.get('nei', 0.0):.3f}"
                )
                print(ev.text[:150])
                
        print("-" * 80)
        
        return is_correct

    except Exception as e:
        print(f" [{current_idx}/{total_count}] NLI Pipeline Error: {str(e)}")
        traceback.print_exc()
        return False


async def test_nli_accuracy(samples: int = 200, batch_size: int = 2):
    print("[INIT] Loading FEVER v1.0 dataset...")
    dataset = load_dataset("fever", "v1.0")
    dev_set = dataset["labelled_dev"]
    
    # Shuffle dataset for rich samples
    raw_valid = [s for s in dev_set if s["label"] in ["SUPPORTS", "REFUTES"]]
    random.seed(42)  # Keep results consistent across debug runs
    random.shuffle(raw_valid)
    valid_samples = raw_valid[:samples]
    total_samples = len(valid_samples)
    
    print(f"[START] Activating DeBERTa NLI evaluation on {total_samples} samples...")
    print("=" * 85)
    
    start_time = time.time()
    nli_hit = 0
    total_valid_executed = 0
    
    for i in range(0, total_samples, batch_size):
        batch = valid_samples[i:i + batch_size]
        tasks = [process_nli_sample(sample, i + idx + 1, total_samples) for idx, sample in enumerate(batch)]
        batch_results = await asyncio.gather(*tasks)
        
        for res in batch_results:
            if res is not None:
                total_valid_executed += 1
                if res is True:
                    nli_hit += 1
                    
        await asyncio.sleep(0.3)
        sys.stdout.flush()

    end_time = time.time()
    print("\n" + "=" * 25 + " DEBERTA NLI ACCURACY REPORT " + "=" * 25)
    print(f"Total execution time             : {end_time - start_time:.2f} seconds")
    print(f"Successfully executed samples    : {total_valid_executed}")
    print(f"Correct DeBERTa predictions      : {nli_hit}")
    if total_valid_executed > 0:
        print(f"FINAL ACCURACY                   : {nli_hit / total_valid_executed * 100:.2f}%")
    print("=" * 79)


if __name__ == "__main__":
    # Updated samples count to 200 as requested
    asyncio.run(test_nli_accuracy(samples=200, batch_size=2))