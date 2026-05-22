# Application constants

VERDICT_TYPES = {
    "REFUTED": 0,
    "DISPUTED": 1,
    "NOT_ENOUGH_INFO": 2,
    "SUPPORTED": 3,
}

STANCE_TYPES = {
    "contradicts": 0,
    "neutral": 1,
    "supports": 2,
}

# Model paths and identifiers
MODELS = {
    "CLIP": "openai/clip-vit-base-patch32",
    "ROBERTA": "roberta-large-mnli",
    "SENTENCE_TRANSFORMER": "sentence-transformers/all-MiniLM-L6-v2",
}

# Default thresholds
THRESHOLDS = {
    "MIN_CONFIDENCE": 0.3,
    "MIN_SIMILARITY": 0.5,
    "MIN_ENTAILMENT": 0.3,
}

# API Response templates
ERROR_RESPONSES = {
    "INVALID_INPUT": "Invalid input provided",
    "MODEL_ERROR": "Model inference error",
    "RETRIEVAL_ERROR": "Evidence retrieval error",
    "INTERNAL_ERROR": "Internal server error",
}
