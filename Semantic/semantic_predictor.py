"""semantic_predictor.py - inference wrapper for the context-aware (semantic) sentiment model."""
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

MAX_CHARS = 300
_CLAUSE_SPLIT = re.compile(
    r"[,;:]|\s[-\u2013\u2014]\s|\b(?:but|while|whereas|although|though|however)\b", flags=re.IGNORECASE
)


def validate_text(text):
    """Return a cleaned headline or raise ValueError with a readable message."""
    if not isinstance(text, str):
        raise ValueError("Headline must be text.")
    text = " ".join(text.split())
    if not text:
        raise ValueError("Headline is empty.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Headline is too long ({len(text)} characters; the limit is {MAX_CHARS}).")
    return text


def split_phrases(text):
    """Split a headline into phrases: clauses at commas / 'but' / 'while' ..., else chunks of 3 words."""
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(text) if p and p.strip()]
    if len(parts) >= 2:
        return parts
    words = text.split()
    if len(words) <= 4:
        return [text.strip()]
    return [" ".join(words[i:i + 3]) for i in range(0, len(words), 3)]


class SemanticSentimentPredictor:
    def __init__(self, artifact_dir="semantic_artifacts"):
        folder = Path(artifact_dir)
        self.meta = json.loads((folder / "model_metadata.json").read_text(encoding="utf-8"))
        local_embedder = folder / "embedder"
        self.embedder = SentenceTransformer(str(local_embedder) if local_embedder.exists()
                                            else self.meta["embedder_model"])
        self.clf = joblib.load(folder / "semantic_classifier.joblib")
        encoder_path = folder / "label_encoder.joblib"
        if encoder_path.exists():
            self.classes_ = [str(c) for c in joblib.load(encoder_path).classes_]
        else:
            self.classes_ = [str(c) for c in self.meta["classes"]]
        self._pos = self.classes_.index("Positive")

    def _proba(self, texts):
        vectors = self.embedder.encode(list(texts), batch_size=64, convert_to_numpy=True,
                                       normalize_embeddings=True, show_progress_bar=False)
        return self.clf.predict_proba(vectors)

    def predict_one(self, text):
        text = validate_text(text)
        proba = self._proba([text])[0]
        return {
            "headline": text,
            "prediction": self.classes_[int(proba.argmax())],
            "confidence": float(proba.max()),
            "probabilities": {c: float(p) for c, p in zip(self.classes_, proba)},
        }

    def predict_labels(self, texts):
        """Fast batch prediction: returns a list of 'Positive' / 'Negative' labels."""
        proba = self._proba([validate_text(t) for t in texts])
        return [self.classes_[int(i)] for i in proba.argmax(axis=1)]

    def explain_one(self, text, max_items=8):
        """Phrase-by-phrase breakdown.

        alone_pos : P(Positive) of the phrase read on its own
        impact    : P(Positive of full headline) - P(Positive with this phrase removed)
                    > 0 means the phrase pushes the headline toward Positive, < 0 toward Negative
        """
        text = validate_text(text)
        parts = split_phrases(text)
        p_full = float(self._proba([text])[0][self._pos])
        if len(parts) == 1:
            return [{"order": 0, "phrase": parts[0], "alone_pos": p_full, "impact": p_full - 0.5,
                     "pushes_toward": "Positive" if p_full >= 0.5 else "Negative"}]
        without = [" ".join(parts[:i] + parts[i + 1:]) for i in range(len(parts))]
        probs = self._proba(parts + without)[:, self._pos]
        alone, removed = probs[:len(parts)], probs[len(parts):]
        items = []
        for i, (phrase, a, r) in enumerate(zip(parts, alone, removed)):
            impact = p_full - float(r)
            items.append({"order": i, "phrase": phrase, "alone_pos": float(a), "impact": impact,
                          "pushes_toward": "Positive" if impact >= 0 else "Negative"})
        items.sort(key=lambda d: abs(d["impact"]), reverse=True)
        return items[:max_items]
