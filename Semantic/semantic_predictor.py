"""semantic_predictor.py — Inference & phrase-level explanation engine for the context-aware sentiment model.

This module shares the exact preprocessing, Transformer backbone, and PyTorch MLP head trained in
news-headline-semantic.ipynb so both the notebook and the Tkinter desktop app (app_tk.py) produce
identical predictions and linguistic phrase-by-phrase explanations.
"""

import json  # Imports json to load model_metadata.json describing the saved architecture and classes.
import re  # Imports regular expressions for whitespace normalization and linguistic phrase segmentation.
from pathlib import Path  # Imports Path for clean, cross-platform artifact directory resolution.

import joblib  # Imports joblib to load label_encoder.joblib (and legacy scikit-learn classifiers if present).
import numpy as np  # Imports NumPy for probability array math and numerical utilities.
import torch  # Imports PyTorch to run fast offline inference with the fine-tuned Transformer + MLP head.
import torch.nn as nn  # Imports PyTorch neural network layers needed to reconstruct the saved MLP head.
from transformers import AutoModel, AutoTokenizer  # Imports Hugging Face classes to reload the offline encoder.

MAX_CHARS = 300  # Maximum accepted headline length to protect the UI from pasted walls of text.

# ---- Linguistic Clause & Phrase Segmentation Patterns (Task 11C) ----
# Splits on structural punctuation (commas, semicolons, colons, dashes) while preserving apostrophes and negation.
_PUNCT_SPLIT = re.compile(r"\s*(?:[,;:]|\s+[-\u2013\u2014]\s+)\s*")
# Splits before subordinating contrast/cause words while KEEPING the marker attached (e.g., "despite the crisis").
_KEEP_PREFIX_CONJ = re.compile(
    r"\s+(?=(?:despite|in\s+spite\s+of|although|though|whereas|while|because|since|amid)\b)",
    flags=re.IGNORECASE,
)
# Splits on coordinating contrast conjunctions ('but', 'however', 'yet') unless part of "not only ... but ...".
_COORD_SPLIT = re.compile(r"\s+(?:but|however|yet)\s+", flags=re.IGNORECASE)
# Removes a redundant leading coordinating word (like "but ") when punctuation already broke the clause before it.
_LEADING_COORD = re.compile(r"^(?:but|however|yet)\s+", flags=re.IGNORECASE)
# Secondary temporal/causal splitters used only when a single clause is still long (>= 6 words).
_SECONDARY_CLAUSE = re.compile(r"\s+(?=(?:as|after|before|when|if|unless|until)\b)", flags=re.IGNORECASE)

# Grammatical glue words and negation/verb leaders that must NEVER be separated from the word that follows them.
_PROTECTED_LEADERS = {
    "not", "no", "never", "none", "neither", "nor", "without", "hardly", "barely",
    "despite", "although", "though", "while", "whereas", "amid", "against", "under",
    "over", "for", "of", "to", "in", "on", "at", "by", "from", "with", "into",
    "a", "an", "the", "this", "that", "these", "those", "its", "their", "his", "her",
    "very", "far", "more", "most", "less", "only", "still", "just", "almost",
    "is", "are", "was", "were", "did", "does", "do", "has", "have", "had", "will",
    "would", "could", "should", "can", "may", "might", "be", "been", "being",
    "fail", "fails", "failed", "failing", "unable", "refuse", "refuses", "refused",
    "remains", "remain", "continues", "continue", "longer",
}


def validate_text(text: str) -> str:
    """Validate and lightly normalize a user-supplied headline while preserving punctuation and negation.

    Args:
        text: Raw headline string typed by a user or loaded from a dataset.

    Returns:
        Cleaned headline string with normalized whitespace.

    Raises:
        ValueError: If input is not a string, is empty after trimming, or exceeds MAX_CHARS.
    """
    if not isinstance(text, str):  # Rejects None, numbers, lists, or other non-text inputs cleanly.
        raise ValueError("Headline must be text.")  # Raises a readable validation error for the UI.
    text = text.replace("\xa0", " ")  # Converts non-breaking spaces into standard spaces (matches notebook cleaning).
    text = re.sub(r"\s+", " ", text).strip()  # Collapses repeated whitespace and trims ends without dropping negation/punctuation.
    if not text:  # Rejects empty or whitespace-only inputs.
        raise ValueError("Headline is empty.")  # Returns clear guidance to the user.
    if len(text) > MAX_CHARS:  # Guards against overly long paragraphs pasted into the headline box.
        raise ValueError(f"Headline is too long ({len(text)} characters; the limit is {MAX_CHARS}).")  # Reports exact limit.
    return text  # Returns clean, context-preserving headline text.


def _clean_phrase_edge(segment: str) -> str:
    """Trim stray separator punctuation from the start/end of an extracted phrase while keeping inner punctuation."""
    return re.sub(r"\s+", " ", segment).strip(" ,;:-–—")  # Removes dangling boundary separators cleanly.


def _smart_chunk_long_clause(clause: str, target_words: int = 4) -> list[str]:
    """Break a long single clause (>= 7 words) into natural phrases without splitting negation or verb/prep expressions."""
    words = clause.split()  # Splits clause into whitespace-separated tokens.
    if len(words) <= 5:  # Keeps short clauses intact so expressions like "not bad" or "failed to deliver" stay whole.
        return [clause]  # Returns single intact phrase.

    chunks = []  # Stores completed natural phrase chunks.
    current = []  # Accumulates tokens belonging to the current phrase.
    for i, word in enumerate(words):  # Walks sequentially through every word in the clause.
        current.append(word)  # Appends current word to the active phrase buffer.
        remaining = len(words) - (i + 1)  # Counts how many words still follow this position.
        if remaining == 0:  # Stops checking split boundaries once at the last word.
            break  # Exits loop cleanly.

        w_low = re.sub(r"[^a-z']", "", word.lower())  # Extracts lowercase alphabetic form of current word.
        next_low = re.sub(r"[^a-z']", "", words[i + 1].lower())  # Extracts lowercase alphabetic form of the upcoming word.

        # Avoid splitting right after negation, auxiliary verbs, determiners, or inside idioms like "no longer" / "failed to"
        avoid_split = (
            remaining < 2  # Never strand a meaningless 1-word tail at the end of a headline.
            or w_low in _PROTECTED_LEADERS  # Never separate "not", "no", "failed", "despite", etc. from what they modify.
            or w_low.endswith("n't")  # Never break contractions like "didn't" or "isn't" away from their verb.
            or next_low in {"to", "longer", "only", "enough", "from"}  # Keeps multi-word idioms intact.
        )

        # Prefer splitting right BEFORE a new prepositional/negation modifier once current phrase has >= 3 words
        natural_break = (
            len(current) >= 3
            and remaining >= 2
            and next_low in {"not", "no", "never", "without", "despite", "against", "amid", "for", "from", "in", "on", "over", "after", "before", "as", "to"}
            and w_low not in _PROTECTED_LEADERS
            and not w_low.endswith("n't")
        )

        if (len(current) >= target_words and not avoid_split) or natural_break:  # Commits phrase at a safe linguistic boundary.
            chunks.append(" ".join(current))  # Joins accumulated words back into a readable phrase.
            current = []  # Resets buffer for the next phrase.

    if current:  # Flushes any remaining words at the end of the clause.
        if chunks and len(current) == 1:  # Merges a stray 1-word ending onto the previous phrase.
            chunks[-1] = chunks[-1] + " " + current[0]  # Keeps final phrase semantically complete.
        else:  # Appends final multi-word phrase normally.
            chunks.append(" ".join(current))  # Adds final phrase to list.
    return chunks  # Returns list of negation-safe linguistic phrases.


def split_phrases(text: str) -> list[str]:
    """Segment a headline into meaningful semantic clauses and phrases while preserving negation and contrast cues.

    Unlike naive fixed 3-word chunking, this splitter:
      1. Separates clauses at punctuation (, ; : - —) and contrast words (but, although, despite, while, whereas).
      2. Preserves expressions like 'not bad', 'not good', 'no longer', 'failed to deliver', 'despite the crisis',
         and 'not only ... but ...' as intact semantic units.
      3. Only chunks single clauses that are 7+ words long, respecting negation and grammatical boundaries.
    """
    text = validate_text(text)  # Ensures clean normalized spacing before segmentation.

    # Step 1: Split along explicit punctuation clause boundaries (, ; : —)
    stage1 = [_clean_phrase_edge(p) for p in _PUNCT_SPLIT.split(text) if _clean_phrase_edge(p)]

    # Step 2: Split on coordinating contrast words ('but', 'however', 'yet'), keeping "not only ... but ..." unified
    stage2 = []  # Stores clauses after coordinating-conjunction splitting.
    for seg in stage1:  # Inspects each punctuation-separated segment.
        if re.search(r"\bnot\s+only\b.*\bbut\b", seg, flags=re.IGNORECASE):  # Protects correlative "not only X but Y".
            stage2.append(seg)  # Keeps the entire correlative construction together.
            continue  # Moves to next segment.
        seg_no_lead = _LEADING_COORD.sub("", seg).strip()  # Strips leading "but " after comma (e.g., ", but the economy...").
        sub_parts = [_clean_phrase_edge(p) for p in _COORD_SPLIT.split(seg_no_lead) if _clean_phrase_edge(p)]  # Splits inner "but".
        stage2.extend(sub_parts if sub_parts else [seg])  # Appends resulting clauses cleanly.

    # Step 3: Split before subordinating contrast/cause markers while KEEPING the marker attached to its clause
    stage3 = []  # Stores clauses with subordinating markers preserved (e.g., "despite the crisis").
    for seg in stage2:  # Inspects each stage-2 segment.
        sub_parts = [_clean_phrase_edge(p) for p in _KEEP_PREFIX_CONJ.split(seg) if _clean_phrase_edge(p)]  # Lookahead split.
        if len(sub_parts) >= 2 and all(len(p.split()) >= 2 for p in sub_parts):  # Accepts split only when every piece has >= 2 words.
            stage3.extend(sub_parts)  # Keeps both meaningful clauses.
        else:  # Leaves segment intact if splitting would create a tiny fragment.
            stage3.append(seg)  # Appends intact clause.

    # Step 4: If still one clause with >= 6 words, try temporal/causal linkers ('as', 'after', 'before', 'when')
    if len(stage3) == 1 and len(stage3[0].split()) >= 6:  # Applies secondary splitting only to longer single clauses.
        sub_parts = [_clean_phrase_edge(p) for p in _SECONDARY_CLAUSE.split(stage3[0]) if _clean_phrase_edge(p)]  # Splits at 'as'/'after'/etc.
        if len(sub_parts) >= 2 and all(len(p.split()) >= 2 for p in sub_parts):  # Ensures balanced multi-word clauses.
            stage3 = sub_parts  # Updates clause list.

    # Step 5: Apply negation-safe chunking only to clauses that are still 7+ words long
    final_phrases = []  # Final ordered list of extracted semantic phrases.
    for seg in stage3:  # Checks length of every clause.
        if len(seg.split()) >= 7:  # Splits very long clauses cleanly at safe grammatical boundaries.
            final_phrases.extend(_smart_chunk_long_clause(seg))  # Extends with safe sub-phrases.
        else:  # Keeps compact clauses (1–6 words) completely intact.
            final_phrases.append(seg)  # Appends intact clause.

    return final_phrases if final_phrases else [text]  # Guarantees at least one non-empty phrase is always returned.


def _build_mlp_head(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """Reconstruct the exact PyTorch BatchNorm + ReLU + Dropout MLP head saved by the notebook."""
    return nn.Sequential(  # Layer indices match the saved checkpoint state_dict keys exactly.
        nn.Linear(in_dim, hidden_dim),  # Fully connected projection from 384-D embedding to hidden layer.
        nn.BatchNorm1d(hidden_dim),  # Batch Normalization (uses saved running mean/variance during inference).
        nn.ReLU(),  # ReLU non-linear activation.
        nn.Dropout(dropout),  # Dropout layer (automatically inactive during .eval() inference mode).
        nn.Linear(hidden_dim, 1),  # Final binary classification neuron outputting raw sentiment logit.
    )


class SemanticSentimentPredictor:
    """Offline inference and phrase-explanation wrapper for the trained Philippine News Sentiment model."""

    def __init__(self, artifact_dir="semantic_artifacts"):
        """Load saved Transformer backbone, trained PyTorch MLP head (or legacy classifier), and metadata."""
        folder = Path(artifact_dir)  # Resolves artifact folder path cleanly.
        meta_path = folder / "model_metadata.json"  # Path to JSON metadata generated during packaging.
        if not meta_path.exists():  # Fails clearly if the model folder has not been generated yet.
            raise FileNotFoundError(f"Model metadata not found at {meta_path}. Run news-headline-semantic.ipynb first.")

        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))  # Parses saved model configuration and metrics.
        encoder_path = folder / "label_encoder.joblib"  # Path to saved class label encoder.
        if encoder_path.exists():  # Loads exact class ordering from LabelEncoder when available.
            self.classes_ = [str(c) for c in joblib.load(encoder_path).classes_]  # Typically ['Negative', 'Positive'].
        else:  # Falls back to metadata class list if label_encoder.joblib is absent.
            self.classes_ = [str(c) for c in self.meta["classes"]]
        self._pos = self.classes_.index("Positive")  # Stores index of 'Positive' probability column.
        self._neg = self.classes_.index("Negative")  # Stores index of 'Negative' probability column.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically uses GPU if available, else CPU.

        # Load the upgraded PyTorch Partially Fine-Tuned Transformer + MLP Head (v2.0) when present
        pt_encoder_dir = folder / "encoder"  # Offline fine-tuned Hugging Face Transformer directory.
        pt_head_file = folder / "mlp_head.pt"  # Saved PyTorch MLP classification head checkpoint.
        if pt_encoder_dir.exists() and pt_head_file.exists():  # Preferred upgraded architecture path.
            self._backend = "pytorch_transformer_mlp"  # Records active inference backend.
            self.tokenizer = AutoTokenizer.from_pretrained(str(pt_encoder_dir))  # Loads offline subword tokenizer.
            self.encoder = AutoModel.from_pretrained(str(pt_encoder_dir)).to(self.device).eval()  # Loads fine-tuned encoder in eval mode.
            ckpt = torch.load(pt_head_file, map_location=self.device, weights_only=True)  # Safely loads MLP head weights onto target device.
            self._pooling = ckpt.get("pooling", "cls")  # Supports both 'cls_mean' (v3.0) and 'cls' (v2.0) checkpoints cleanly.
            self.head = _build_mlp_head(ckpt["in_dim"], ckpt["hidden_dim"], ckpt["dropout"]).to(self.device)  # Rebuilds MLP structure.
            self.head.load_state_dict(ckpt["state_dict"])  # Loads trained weights and BatchNorm running statistics.
            self.head.eval()  # Disables dropout and locks BatchNorm for deterministic inference.
        else:  # Backward-compatible fallback for legacy v1.0 SentenceTransformer + scikit-learn joblib artifacts.
            from sentence_transformers import SentenceTransformer  # Lazy import only needed for legacy artifacts.
            self._backend = "sklearn_embedder"  # Records legacy inference backend.
            local_embedder = folder / "embedder"  # Legacy local MiniLM folder.
            self.embedder = SentenceTransformer(str(local_embedder) if local_embedder.exists() else self.meta["embedder_model"], device=self.device)
            self.clf = joblib.load(folder / "semantic_classifier.joblib")  # Loads legacy scikit-learn classifier.

    def _proba(self, texts) -> np.ndarray:
        """Compute a 2-column probability matrix [P(Negative), P(Positive)] for a batch of text strings."""
        texts = list(texts)  # Converts any iterable into an indexable Python list.
        if self._backend == "pytorch_transformer_mlp":  # Runs upgraded unified Transformer + PyTorch MLP forward pass.
            p_pos_chunks = []  # Collects Positive probabilities across mini-batches.
            with torch.no_grad():  # Disables gradient tracking for fast, memory-efficient inference.
                for start in range(0, len(texts), 64):  # Encodes up to 64 texts per forward pass.
                    enc = self.tokenizer(  # Tokenizes text batch with padding and truncation.
                        texts[start:start + 64], padding=True, truncation=True, max_length=64, return_tensors="pt"
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}  # Moves tensors to active device.
                    seq = self.encoder(**enc).last_hidden_state  # Contextual token embeddings (batch, seq_len, hidden_size).
                    if self._pooling == "cls_mean":  # Concatenates L2-normalized [CLS] and masked mean-pooled sentence vectors.
                        cls_v = nn.functional.normalize(seq[:, 0, :], p=2, dim=1)
                        mask_f = enc["attention_mask"].unsqueeze(-1).to(seq.dtype)
                        mean_v = nn.functional.normalize((seq * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-6), p=2, dim=1)
                        feat = torch.cat([cls_v, mean_v], dim=1)
                    else:  # Backward-compatible [CLS]-only pooling for v2.0 checkpoints.
                        feat = nn.functional.normalize(seq[:, 0, :], p=2, dim=1)
                    p_pos = torch.sigmoid(self.head(feat).squeeze(-1)).cpu().numpy()  # Applies Sigmoid to raw logit.
                    p_pos_chunks.append(np.atleast_1d(p_pos))  # Guarantees 1D shape even for single-item batches.
            p_pos_all = np.concatenate(p_pos_chunks, axis=0)  # Combines all batch Positive probabilities.
            proba = np.zeros((len(texts), 2), dtype=float)  # Allocates 2-column probability array.
            proba[:, self._pos] = p_pos_all  # Fills Positive probability column.
            proba[:, self._neg] = 1.0 - p_pos_all  # Fills Negative probability column (sums exactly to 1.0).
            return proba  # Returns [P(Negative), P(Positive)] array matching self.classes_ order.

        # Legacy SentenceTransformer + scikit-learn fallback
        vectors = self.embedder.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return self.clf.predict_proba(vectors)  # Returns 2-column class probability matrix.

    def predict_one(self, text: str) -> dict:
        """Predict sentiment label, confidence, and both Positive/Negative probabilities for one headline."""
        clean_text = validate_text(text)  # Validates and normalizes input using exact training preprocessing.
        proba = self._proba([clean_text])[0]  # Runs inference and retrieves [P(Neg), P(Pos)] vector.
        return {  # Returns rich, structured prediction dictionary for notebook and Tkinter UI.
            "headline": clean_text,  # Normalized headline text actually scored by the model.
            "prediction": self.classes_[int(proba.argmax())],  # Winning class ('Positive' or 'Negative').
            "confidence": float(proba.max()),  # Probability of the winning class (in [0.5, 1.0]).
            "probabilities": {c: float(p) for c, p in zip(self.classes_, proba)},  # Explicit Positive & Negative probabilities.
        }

    def predict_labels(self, texts) -> list[str]:
        """Run fast batched sentiment classification and return a list of 'Positive' / 'Negative' labels."""
        cleaned = [validate_text(t) for t in texts]  # Validates every headline in the batch.
        proba = self._proba(cleaned)  # Computes batched probabilities in one efficient pass.
        return [self.classes_[int(i)] for i in proba.argmax(axis=1)]  # Converts winning indices to label strings.

    def explain_one(self, text: str, max_items: int = 8) -> list[dict]:
        """Break a headline into meaningful phrases and estimate each phrase's standalone sentiment and contextual impact.

        Returned fields per phrase:
          - order        : Original left-to-right position in the headline.
          - phrase       : Extracted meaningful phrase (preserving negation & contrast cues).
          - alone_label  : Individual sentiment ('Positive' or 'Negative') of the phrase read alone.
          - alone_pos    : P(Positive) of the phrase read alone.
          - alone_neg    : P(Negative) of the phrase read alone (1 - alone_pos).
          - without_pos  : P(Positive) of the full headline after removing this phrase.
          - impact       : P(Positive of full headline) - P(Positive with phrase removed).
                           Positive impact pushes the full headline toward Positive; negative toward Negative.
          - pushes_toward: 'Positive' if impact >= 0 else 'Negative'.

        Note: Leave-one-phrase-out occlusion is a heuristic explanation of model sensitivity, not causal proof.
        """
        clean_text = validate_text(text)  # Validates and cleans input text identically to predict_one().
        parts = split_phrases(clean_text)  # Segments headline into meaningful negation-safe phrases.
        p_full = float(self._proba([clean_text])[0][self._pos])  # Computes full-headline P(Positive).

        if len(parts) == 1:  # Handles indivisible short headlines (e.g., "Not bad", "No corruption found") cleanly.
            return [{  # Returns single-phrase entry showing full standalone probabilities and shift from 0.50 neutral midpoint.
                "order": 0,
                "phrase": parts[0],
                "alone_label": "Positive" if p_full >= 0.5 else "Negative",
                "alone_pos": p_full,
                "alone_neg": 1.0 - p_full,
                "without_pos": 0.5,
                "impact": p_full - 0.5,
                "pushes_toward": "Positive" if p_full >= 0.5 else "Negative",
            }]

        # Build leave-one-phrase-out versions of the headline to measure contextual contribution
        without_texts = [" ".join(parts[:i] + parts[i + 1:]) for i in range(len(parts))]  # Removes phrase i one at a time.
        batch_probs = self._proba(parts + without_texts)[:, self._pos]  # Scores all individual & occluded texts in one fast batch.
        alone_probs = batch_probs[:len(parts)]  # Standalone P(Positive) for each phrase.
        removed_probs = batch_probs[len(parts):]  # Full-headline P(Positive) when phrase i is removed.

        items = []  # Collects rich explanation records for every phrase.
        for i, (phrase, a_pos, r_pos) in enumerate(zip(parts, alone_probs, removed_probs)):  # Iterates over phrases in sentence order.
            a_pos_f = float(a_pos)  # Converts NumPy float to native Python float.
            r_pos_f = float(r_pos)  # Converts occluded P(Positive) to native Python float.
            impact = p_full - r_pos_f  # Positive delta => phrase pulls full prediction toward Positive; negative => toward Negative.
            items.append({  # Appends complete phrase-level diagnostics.
                "order": i,
                "phrase": phrase,
                "alone_label": "Positive" if a_pos_f >= 0.5 else "Negative",
                "alone_pos": a_pos_f,
                "alone_neg": 1.0 - a_pos_f,
                "without_pos": r_pos_f,
                "impact": impact,
                "pushes_toward": "Positive" if impact >= 0.0 else "Negative",
            })

        if len(items) > max_items:  # If a very long sentence exceeds max_items, keep the highest-impact phrases in reading order.
            items = sorted(items, key=lambda d: abs(d["impact"]), reverse=True)[:max_items]  # Selects most influential phrases.
            items.sort(key=lambda d: d["order"])  # Restores natural left-to-right sentence order.
        return items  # Returns ordered list of phrase explanation dictionaries.
