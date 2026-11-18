
# news_predictor.py - production-style inference wrapper.
from pathlib import Path
import joblib
import pandas as pd
from news_nlp_utils import clean_headline, build_meta_features, META_COLUMNS, validate_headline

class NewsSentimentPredictor:
    def __init__(self, model_path="news_sentiment_artifacts/ph_news_sentiment_pipeline.joblib"):
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)          # load the fitted preprocessing+model pipeline
        self.classes_ = list(self.model.named_steps['model'].classes_)

    def _frame(self, headline: str) -> pd.DataFrame:
        # Rebuild the EXACT same feature structure (text + metadata columns) used at training time.
        headline = validate_headline(headline)               # raises a clear error on bad input (D4)
        df = pd.DataFrame({'text': [clean_headline(headline)]})
        df = pd.concat([df, build_meta_features(pd.Series([headline]))], axis=1)
        return df[['text'] + META_COLUMNS]

    def predict_one(self, headline: str):
        x = self._frame(headline)
        proba = self.model.predict_proba(x)[0]
        pred = self.classes_[int(proba.argmax())]             # class with the highest predicted probability
        return {
            'headline': headline.strip(),
            'prediction': pred,
            'confidence': float(proba.max()),
            'probabilities': {str(c): float(p) for c, p in zip(self.classes_, proba)},
        }
