
# news_nlp_utils.py - reusable helpers for the Philippines news sentiment pipeline.
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# List of engineered numeric/flag columns that get attached to every headline,
# in addition to the TF-IDF text features. This list is imported elsewhere
# (e.g. the ColumnTransformer) so the model always knows exactly which
# metadata columns to expect.
META_COLUMNS = [
    "char_count", "word_count", "unique_word_ratio", "exclamation_count",
    "question_count", "digit_ratio", "uppercase_ratio", "avg_word_length", "has_negative_domain_term"
]

# Words that signal a genuinely negative real-world topic even when the
# headline's wording is calm/factual (e.g. "Inflation hits new high" has no
# emotionally-loaded words, so TextBlob alone would miss it).
NEGATIVE_DOMAIN_TERMS = [
    'typhoon', 'flood', 'earthquake', 'landslide', 'disaster', 'evacuate',
    'corrupt', 'corruption', 'bribery', 'scandal', 'fraud', 'graft',
    'inflation', 'recession', 'layoff', 'unemployment', 'debt', 'deficit',
    'crash', 'accident', 'death', 'killed', 'dead', 'injured', 'victim',
    'protest', 'unrest', 'violence', 'crime', 'arrest', 'shooting',
    'congestion', 'delay', 'shortage', 'outbreak', 'lockdown', 'ban'
]
# Join the word list into a single regex pattern: word1|word2|word3|...
# so a single re.search() call can check for ANY of them at once.
_NEG_PATTERN = '|'.join(NEGATIVE_DOMAIN_TERMS)


def clean_headline(text: str) -> str:
    # Safe text normalization; keeps negation words and useful punctuation cues.
    # If text is missing (None), replace it with an empty string instead of crashing.
    text = "" if text is None else str(text)
    # Replace non-breaking space characters with normal spaces.
    text = text.replace("\xa0", " ")
    # Collapse any URL (http/https/www links) down to a single "URL" placeholder token,
    # since the exact link text carries no sentiment signal.
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    # Collapse repeated whitespace into single spaces, trim the ends, and lowercase
    # everything so "Rises" and "rises" are treated as the same word by TF-IDF.
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def build_meta_features(series: pd.Series) -> pd.DataFrame:
    # Create deterministic, row-wise headline metadata from raw text (used both
    # as extra model features and for several EDA visualizations above).
    rows = []
    # fillna("") handles any missing headlines safely; astype(str) guarantees
    # every value is a plain string before we start measuring it.
    for raw in series.fillna("").astype(str):
        # Split into individual words on whitespace.
        words = raw.split()
        n_words = len(words)
        # Precompute the lowercase version once, reused below for the keyword check.
        raw_lower = raw.lower()
        rows.append({
            # Total number of characters in the headline.
            "char_count": len(raw),
            # Total number of words in the headline.
            "word_count": n_words,
            # Fraction of words that are unique (max(n_words, 1) avoids divide-by-zero
            # on an empty headline).
            "unique_word_ratio": len(set(w.lower() for w in words)) / max(n_words, 1),
            # Count of "!" characters -- a rough proxy for sensationalized tone.
            "exclamation_count": raw.count("!"),
            # Count of "?" characters.
            "question_count": raw.count("?"),
            # Fraction of characters that are digits (e.g. dates, statistics).
            "digit_ratio": sum(ch.isdigit() for ch in raw) / max(len(raw), 1),
            # Fraction of characters that are uppercase (e.g. ALL-CAPS headlines).
            "uppercase_ratio": sum(ch.isupper() for ch in raw) / max(len(raw), 1),
            # Average word length; 0.0 for an empty headline instead of crashing on
            # an empty mean.
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0.0,
            # 1 if the headline contains ANY of the negative-domain keywords, else 0.
            # bool(...) turns the regex match object (or None) into True/False,
            # and int(...) turns that into 1/0 so it behaves like the other numeric columns.
            "has_negative_domain_term": int(bool(re.search(_NEG_PATTERN, raw_lower))),
        })
    # Turn the list of per-row dictionaries into one DataFrame, keeping the same
    # row index as the input Series so it lines up correctly when concatenated later.
    return pd.DataFrame(rows, index=series.index)


class Winsorizer(BaseEstimator, TransformerMixin):
    # Clip numeric features to training-set quantile bounds (A3): learns the
    # bounds from fit() (training data only) and clips with them in transform().
    def __init__(self, lower=0.01, upper=0.99):
        # Lower/upper quantile cutoffs, e.g. 1st and 99th percentile by default.
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        # Force X into a float array so quantile math works cleanly.
        X = np.asarray(X, dtype=float)
        # Learn the clipping bounds ONLY from the data passed in here (training data),
        # so no information about the test set leaks into these thresholds.
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self  # scikit-learn convention: fit() must return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        # Clamp every value into [lower_, upper_] using the bounds learned in fit().
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        # Clipping doesn't add/remove/rename columns, so output names = input names.
        # Without this method, scikit-learn's pipeline machinery raises an error
        # once a Winsorizer is used inside it -- this is the fix for that.
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        # Fallback if no names were given: just number the columns x0, x1, ...
        n = getattr(self, "lower_", np.array([])).shape[0]
        return np.asarray([f"x{i}" for i in range(n)], dtype=object)


def validate_headline(text: str, max_chars: int = 500) -> str:
    # Reject a missing (None) headline outright.
    if text is None:
        raise ValueError("Headline cannot be empty.")
    # Trim whitespace, then reject a headline that's blank after trimming.
    text = str(text).strip()
    if not text:
        raise ValueError("Headline cannot be empty.")
    # Reject headlines that are unreasonably long (guards the live prediction UI
    # against pasted-in walls of text).
    if len(text) > max_chars:
        raise ValueError(f"Headline is too long. Maximum supported length is {max_chars} characters.")
    return text
