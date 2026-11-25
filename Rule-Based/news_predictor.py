# news_predictor.py - production-style inference & explainability wrapper for Rule-Based sentiment model.
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from news_nlp_utils import clean_headline, build_meta_features, META_COLUMNS, validate_headline


def _patch_sklearn_compat(est):
    """Ensure pickled sklearn 1.7 pipelines work seamlessly when loaded under newer sklearn (e.g., 1.8/1.9+)."""
    # SimpleImputer in sklearn >= 1.8 expects '_fill_dtype' (renamed/added after 1.7's '_fit_dtype')
    if est.__class__.__name__ == "SimpleImputer" and not hasattr(est, "_fill_dtype"):
        est._fill_dtype = getattr(est, "_fit_dtype", getattr(getattr(est, "statistics_", None), "dtype", np.dtype("float64")))
    # Recurse into Pipeline steps and ColumnTransformer sub-transformers
    if hasattr(est, "steps"):
        for _, step in est.steps:
            _patch_sklearn_compat(step)
    if hasattr(est, "transformers_"):
        for _, trans, _ in est.transformers_:
            if trans not in ("drop", "passthrough"):
                _patch_sklearn_compat(trans)
    if hasattr(est, "transformers"):
        for _, trans, _ in est.transformers:
            if trans not in ("drop", "passthrough"):
                _patch_sklearn_compat(trans)
    return est


class NewsSentimentPredictor:
    """Offline inference and feature-contribution wrapper for the TF-IDF + Logistic Regression pipeline."""

    def __init__(self, model_path="news_sentiment_artifacts/ph_news_sentiment_pipeline.joblib"):
        self.model_path = Path(model_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = _patch_sklearn_compat(joblib.load(self.model_path))  # Loads & patches pipeline cleanly.
        self.classes_ = list(self.model.named_steps["model"].classes_)

    def _frame(self, headline: str) -> pd.DataFrame:
        # Rebuild the EXACT same feature structure (text + metadata columns) used at training time.
        headline = validate_headline(headline)  # Raises clear ValueError on empty / overly long input (D4).
        df = pd.DataFrame({"text": [clean_headline(headline)]})
        df = pd.concat([df, build_meta_features(pd.Series([headline]))], axis=1)
        return df[["text"] + META_COLUMNS]

    def predict_one(self, headline: str) -> dict:
        """Predict sentiment label, confidence, and per-class probabilities for a single headline."""
        x = self._frame(headline)
        proba = self.model.predict_proba(x)[0]
        pred = self.classes_[int(proba.argmax())]  # Class with highest predicted probability.
        return {
            "headline": headline.strip(),
            "prediction": pred,
            "confidence": float(proba.max()),
            "probabilities": {str(c): float(p) for c, p in zip(self.classes_, proba)},
        }

    def explain_one(self, headline: str, top_k: int = 10) -> list[dict]:
        """Return the top active TF-IDF terms & metadata features driving the Logistic Regression log-odds."""
        x = self._frame(headline)
        prep = self.model.named_steps.get("prep")
        sel = self.model.named_steps.get("select")
        clf = self.model.named_steps.get("model")
        if prep is None or clf is None or not hasattr(clf, "coef_"):
            return []

        names = prep.get_feature_names_out()
        vec = prep.transform(x)
        if sel is not None:
            names = names[sel.get_support()]
            vec = sel.transform(vec)

        arr = vec.toarray().ravel() if hasattr(vec, "toarray") else np.asarray(vec).ravel()
        coefs = clf.coef_.ravel()
        # Ensure positive sign consistently means pushing toward 'Positive' class
        sign = 1.0 if (len(self.classes_) >= 2 and str(self.classes_[1]) == "Positive") else -1.0
        coefs = coefs * sign
        contribs = arr * coefs

        nz = np.flatnonzero(np.abs(contribs) > 1e-6)
        top_idx = nz[np.argsort(np.abs(contribs[nz]))[::-1][:top_k]]

        items = []
        for idx in top_idx:
            raw_name = str(names[idx])
            if raw_name.startswith("word__"):
                kind, clean_name = "TF-IDF Word/Phrase", raw_name[len("word__"):]
            elif raw_name.startswith("meta__"):
                kind, clean_name = "Headline Metadata", raw_name[len("meta__"):].replace("_", " ")
            else:
                kind, clean_name = "Feature", raw_name
            c_val = float(contribs[idx])
            items.append({
                "feature": clean_name,
                "kind": kind,
                "value": float(arr[idx]),
                "weight": float(coefs[idx]),
                "contribution": c_val,
                "pushes_toward": "Positive" if c_val >= 0.0 else "Negative",
            })
        return items
