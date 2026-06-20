# CrossCheck Extension

A browser extension for automatic fact-checking using hybrid retrieval and natural language inference.

---

## Motivation

Misinformation spreads rapidly on the internet.

This project explores how retrieval techniques and natural language inference can be combined to provide lightweight fact-checking directly inside the browser.

The goal is not only to build a usable browser extension, but also to investigate how retrieval quality affects downstream fact-checking performance.

---

## Features

* Highlight text directly on webpages.
* Retrieve supporting evidence from Wikipedia.
* Hybrid retrieval using lexical and semantic search.
* Cross-Encoder evidence reranking.
* Sentence-level evidence selection.
* NLI-based verdict prediction.
* Display verdicts inside the browser popup.
* Provide evidence sentences and confidence scores.
* Cache previous claims to reduce repeated inference.
* Optional LLM inference for further analysis and explanation.

---

## System Architecture

```text
User Selection
      ↓
Browser Extension
      ↓
FastAPI Backend
      ↓
Topic Extraction
      ↓
Hybrid Retrieval
      ↓
Cross-Encoder Ranking
      ↓
Sentence Selection
      ↓
DeBERTa NLI
      ↓
Verdict + Evidence
```

---

## Project Structure

```text
crosscheck-extension/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── pipelines/
│   │   ├── services/
│   │   ├── schemas/
│   │   └── main.py
│   │
│   ├── tests/
│   │   ├── retrieval_test.py
│   │   └── nli_test.py
│   │
│   └── requirements.txt
│
├── extension/
│   ├── src/
│   │   ├── popup/
│   │   ├── background/
│   │   ├── content/
│   │   ├── services/
│   │   ├── types/
│   │   └── utils/
│   │
│   └── package.json
│
├── ml/
│   ├── evaluation/
│   ├── experiments/
│   └── kaggle/
│
└── README.md
```

---

## Fact-Checking Pipeline

1. Topic Extraction
2. Entity Linking
3. Hybrid Retrieval
4. Cross-Encoder Ranking
5. Sentence Selection
6. NLI Prediction
7. Confidence Calibration
8. Claim Caching

---

## Experiments

Several retrieval and inference strategies were investigated during development.

### Retrieval Experiments

* BM25 retrieval
* Dense retrieval
* Entity-based retrieval
* Hybrid retrieval
* Graph-based retrieval

### Evaluation Tasks

* Recall@k evaluation
* Mean Reciprocal Rank (MRR)
* End-to-end fact-check accuracy
* Evidence quality analysis

### Experimental Environment

Experiments and benchmarks are executed independently inside the `ml/` directory and Kaggle notebooks.

---

## Evaluation

### Model Evaluation

The NLI component was evaluated independently on the FEVER v1.0 dataset.

| Label | Precision | Recall | F1 |
|------|----------:|-------:|---:|
| SUPPORTS | 0.81 | 0.88 | 0.84 |
| REFUTES | 0.70 | 0.69 | 0.69 |
| NOT_ENOUGH_INFO | 0.81 | 0.75 | 0.78 |

| Overall Metric | Score |
|---------------|------:|
| Accuracy | 0.77 |
| Macro F1 | 0.77 |
| Weighted F1 | 0.77 |

Dataset: FEVER v1.0  
Evaluation samples: 19,998

### Retrieval Performance

| Metric   | Score |
| -------- | ----- |
| Recall@1 | 0.69  |
| Recall@3 | 0.84  |
| Recall@5 | 0.86  |
| MRR      | 0.75  |

### End-to-End Performance

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.41  |
| F1 Score | 0.395 |

---

## Installation

### Backend

```bash
`git clone https://github.com/24127382/CrossCheck-Extension`

cd backend

pip install -r requirements.txt

uvicorn app.main:app --reload
```

### Extension

```bash
cd extension

npm install

npm run build
```

Load the generated extension inside the browser developer mode.

---

## Limitations

* Retrieval quality strongly affects final predictions.
* Wikipedia may not contain evidence for recent events.
* Multi-hop reasoning is not supported.
* Confidence calibration remains imperfect.
* The system is currently limited to English claims.

---

## Future Work

* Larger retrieval corpora.
* Multi-hop retrieval.
* Knowledge graph integration.
* LLM-based evidence summarization.
* Human feedback loop.
* Deployment and scalability improvements.
* Better confidence calibration.

---
