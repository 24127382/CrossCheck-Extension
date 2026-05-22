# Project Structure Documentation

## Architecture Overview

This is a **Fact-Checking Browser Extension** with three main components:

### 1. Extension (Frontend)
Browser extension that detects selected text and sends fact-check requests.

```
extension/
├── manifest.json          # Extension configuration
├── package.json          # Node dependencies
├── vite.config.ts        # Build configuration
├── public/               # Static assets
├── src/
│   ├── popup/           # Popup UI & results display
│   ├── content/         # Content script (webpage injection)
│   ├── background/      # Service worker (API calls, auth)
│   ├── services/        # API, factcheck, storage services
│   ├── types/           # TypeScript types
│   └── utils/           # Helper functions
```

**Flow**: User selects text → Extension detects → Sends to backend → Displays results

### 2. Backend (API)
FastAPI server that orchestrates the fact-checking pipeline.

```
backend/
├── app/
│   ├── api/             # Route handlers
│   │   └── routes/      # factcheck, health, debug endpoints
│   ├── core/            # Configuration, logging, constants
│   ├── services/        # Orchestration layer
│   │   ├── retrieval_service.py     # Find evidence
│   │   ├── entailment_service.py    # NLI inference
│   │   ├── llm_service.py           # Summarization
│   │   ├── clip_service.py          # Multimodal
│   │   └── ranking_service.py       # Evidence ranking
│   ├── pipelines/       # factcheck_pipeline.py (main flow)
│   ├── models/          # Model wrappers
│   ├── schemas/         # Pydantic models
│   └── main.py          # App entry point
├── requirements.txt
├── Dockerfile
└── tests/
```

**Pipeline Flow**:
```
Claim → Retrieve Evidence → Encode → Compute Entailment 
→ Rank → Generate Verdict → Summarize → Response
```

**Response Format**:
```json
{
  "claim": "...",
  "verdict": "REFUTED|SUPPORTED|NOT_ENOUGH_INFO|DISPUTED",
  "confidence": 0.82,
  "summary": "...",
  "evidences": [
    {"source": "Reuters", "stance": "contradicts", "score": 0.91}
  ]
}
```

### 3. ML (Research & Development)
Separate folder for experimentation and model evaluation.

```
ml/
├── experiments/         # Different approaches
│   ├── fever_baseline/
│   ├── entailment_eval/
│   └── clip_similarity/
├── notebooks/          # Jupyter analysis
├── evaluation/         # Metrics & benchmarking
├── datasets/           # Training data
├── checkpoints/        # Model weights
└── training/           # Training scripts
```

### 4. Infrastructure
Deployment and orchestration configs.

```
infra/
├── docker/        # Service Dockerfiles
├── compose/       # docker-compose.yml
├── nginx/         # Reverse proxy config
└── mlflow/        # ML tracking
```

### 5. Data
Data management and caching.

```
data/
├── raw/           # Original datasets
├── processed/     # Cleaned data
├── cache/         # Runtime cache
└── demo/          # Test data
```

## Key Design Decisions

✅ **Separation of concerns**: Extension, backend, and ML are independently scalable
✅ **Modular services**: Each step in the pipeline is a separate service
✅ **Clean schema**: Pydantic models ensure data consistency
✅ **Easy evaluation**: ML folder is decoupled from production
✅ **Model flexibility**: Easy to swap models without breaking the pipeline
✅ **Clear API contract**: Consistent response format for UI and evaluation
