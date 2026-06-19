import asyncio
import urllib.parse
import traceback

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

from app.services.retrieval_service import (
    retrieval_service,
    query_builder
)

from app.services.ranking_service import (
    ranking_service,
    sentence_selector
)

try:
    from app.services.retrieval_service import entity_linker
    import spacy

    nlp = spacy.load("en_core_web_sm")
except:
    entity_linker = None
    nlp = None


# ==========================================================
# Semantic Similarity Model
# ==========================================================

sim_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================================
# Utils
# ==========================================================

def clean_wiki_title(title: str):

    if not title:
        return ""

    if "wikipedia.org/wiki/" in title:
        title = title.split(
            "wikipedia.org/wiki/"
        )[-1]

    title = urllib.parse.unquote(title)

    return (
        title
        .strip()
        .replace(" ", "_")
        .lower()
    )


# ==========================================================
# Load Gold Sentence
# ==========================================================

def get_gold_sentence_text(
    sample,
    retrieved_docs
):
    """
    Cố gắng lấy đúng gold sentence từ document
    mà retrieval vừa tải về.
    """

    gold_page = clean_wiki_title(
        sample["evidence_wiki_url"]
    )

    gold_sid = int(
        sample["evidence_sentence_id"]
    )

    for doc in retrieved_docs:

        page = clean_wiki_title(
            getattr(doc, "source", "")
        )

        if page != gold_page:
            continue

        sentences = sentence_selector.clean_and_split(
            doc.text
        )

        if gold_sid < len(sentences):
            return sentences[gold_sid]

    return None


# ==========================================================
# Single Sample
# ==========================================================

async def evaluate_sample(sample):
    print("\n" + "=" * 100)
    print("[START]")
    
    claim = sample["claim"]
    print(f"DEBUG: Processing claim: {claim[:120]}")

    try:

        # ----------------------------------
        # Entity Linking
        # ----------------------------------
        print("DEBUG: Extracting topics and linking entities...")
        topics = query_builder.extract_topics(
            claim
        )

        entity_links = []

        if nlp and entity_linker:

            doc = nlp(claim)

            mentions = entity_linker.extract_mentions(
                doc
            )

            for m in mentions:

                linked = entity_linker.link(m)

                if linked:
                    entity_links.append(
                        linked
                    )

        # ----------------------------------
        # Retrieval
        # ----------------------------------
        print("DEBUG: Starting retrieval step")

        docs = retrieval_service.retrieve(
            topics=topics,
            entity_links=entity_links
        )

        if not docs:
            return None


        print(
            f"Retrieved docs: {len(docs)}"
        )

        for i, d in enumerate(docs[:5]):

            page = getattr(
                d,
                "source",
                "unknown"
            )

            print(
                f"  {i+1}. {page}"
            )
        # ----------------------------------
        # Gold sentence
        # ----------------------------------

        gold_sentence = get_gold_sentence_text(
            sample,
            docs
        )

        if not gold_sentence:
            print("DEBUG: Could not find gold sentence in retrieved docs")
            return None
        print(f"DEBUG: Gold sentence found: {gold_sentence[:100]}")
        # ----------------------------------
        # Rank Docs
        # ----------------------------------
        print("DEBUG: Ranking documents...")
        ranked_docs = await ranking_service.rank_evidence(
            claim,
            docs
        )

        ranked_docs = sorted(
            ranked_docs,
            key=lambda x: x.final_score,
            reverse=True
        )

        top_docs = [
            x.evidence
            for x in ranked_docs[:5]
        ]
        print(f"DEBUG: Top {len(top_docs)} documents after ranking")
        # ----------------------------------
        # Sentence Selection
        # ----------------------------------
        print("DEBUG: Selecting sentences from ranked docs...")
        selected_sentences = sentence_selector.select(
            claim=claim,
            evidences=top_docs,
            top_n=3
        )

        if not selected_sentences:
            print("DEBUG: No sentences selected")
            return None

        pred_texts = [
            ev.text
            for ev in selected_sentences
        ]
        print(f"DEBUG: Selected {len(pred_texts)} sentences:")
        for idx, txt in enumerate(pred_texts):
            print(f"DEBUG: Selected sentence {idx+1}: {txt[:150]}")
        # ----------------------------------
        # Semantic Similarity
        # ----------------------------------

        gold_emb = sim_model.encode(
            gold_sentence,
            convert_to_tensor=True
        )

        pred_embs = sim_model.encode(
            pred_texts,
            convert_to_tensor=True
        )

        scores = util.cos_sim(
            gold_emb,
            pred_embs
        )[0]

        best_score = float(
            scores.max().item()
        )

        semantic_hit = (
            best_score >= 0.70
        )
        print(f"DEBUG: Semantic similarity score: {best_score}")
        return {
            "hit": semantic_hit,
            "score": best_score,
            "claim": claim,
            "gold": gold_sentence,
            "predicted": pred_texts
        }

    except Exception as e:

        print(
            f"[ERROR] {str(e)}"
        )

        traceback.print_exc()

        return None


# ==========================================================
# Full Evaluation
# ==========================================================

async def run_test(
    max_samples=100
):

    dataset = load_dataset(
        "fever",
        "v1.0"
    )["labelled_dev"]

    total = 0

    semantic_hits = 0

    similarity_sum = 0.0

    fail_cases = []

    for sample in dataset:

        if sample["label"] == "NOT ENOUGH INFO":
            continue

        result = await evaluate_sample(
            sample
        )

        if result is None:
            continue

        total += 1

        similarity_sum += result["score"]

        if result["hit"]:
            semantic_hits += 1

        else:

            fail_cases.append(
                result
            )

        if total >= max_samples:
            break

    print("\n")
    print("=" * 80)

    print(
        f"Total Samples: {total}"
    )

    print(
        f"Semantic Recall@3: "
        f"{semantic_hits}/{total}"
        f" = {semantic_hits/total:.4f}"
    )

    print(
        f"Average Similarity: "
        f"{similarity_sum/total:.4f}"
    )

    print("=" * 80)

    print("\nTop Fail Cases\n")

    for case in fail_cases[:10]:

        print("-" * 80)

        print(
            "CLAIM:",
            case["claim"]
        )

        print(
            "\nGOLD:"
        )

        print(
            case["gold"]
        )

        print(
            "\nSELECTED:"
        )

        for s in case["predicted"]:

            print("-", s)

        print(
            "\nBEST SCORE:",
            round(
                case["score"],
                4
            )
        )

        print("-" * 80)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    asyncio.run(
        run_test(
            max_samples=100
        )
    )