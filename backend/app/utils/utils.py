from sentence_transformers import  util

def cosine_score(model, claim, text):
    c = model.encode(claim, convert_to_tensor=True)
    t = model.encode(text, convert_to_tensor=True)
    return util.cos_sim(c, t).item()


def bm25_score(bm25, claim_tokens, doc_tokens):
    return bm25.get_score(claim_tokens, doc_tokens)


def entity_match(claim_ents, doc_ents):
    if not claim_ents:
        return 0.0
    return len(set(claim_ents) & set(doc_ents)) / len(set(claim_ents))


def scorer(claim, text, claim_ents, doc_ents, model, bm25):
    return (
        0.5 * cosine_score(model, claim, text) +
        0.3 * bm25_score(bm25, claim.split(), text.split()) +
        0.2 * entity_match(claim_ents, doc_ents)
    )
