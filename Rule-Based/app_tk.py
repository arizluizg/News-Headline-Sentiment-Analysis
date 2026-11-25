"""app_tk.py — Tkinter desktop application for Philippines News Headlines Rule-Based Sentiment Analysis.

Run this application from the Rule-Based/ directory (100% offline, instant startup, built-in Tkinter — no Streamlit needed!):

    python app_tk.py

Powered directly by news_predictor.py + news_sentiment_artifacts/ph_news_sentiment_pipeline.joblib
(TF-IDF unigrams/bigrams + engineered headline metadata + Tuned Logistic Regression).
"""

import csv  # Allows exporting the session prediction history table to CSV.
import datetime as dt  # Timestamps every analyzed headline in the History tab.
import json  # Reads model_metadata.json to display live pipeline details in the About tab.
import queue  # Thread-safe mailbox passing background worker results back to Tkinter's UI loop.
import sys  # Ensures local Rule-Based/ modules (news_predictor, news_nlp_utils) always resolve cleanly.
import threading  # Keeps GUI responsive during model load and prediction.
import time  # Measures real-time inference latency in milliseconds.
import tkinter as tk  # Python's built-in desktop GUI toolkit.
from pathlib import Path  # Cross-platform path handling.
from tkinter import filedialog, messagebox, ttk  # Modern ttk widgets, save dialogs, and alerts.

BASE = Path(__file__).resolve().parent  # Absolute path of the Rule-Based/ folder containing app_tk.py.
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# ---- Accessible High-Contrast Color Palette (matching Semantic/app_tk.py) ----
POS_COLOR = "#1a9e4b"    # Green for Positive sentiment & positive TF-IDF feature weights.
NEG_COLOR = "#d64545"    # Red for Negative sentiment & negative TF-IDF feature weights.
MUTED = "#5f6b7a"        # Slate gray for subtitles, hints, and status bar.
TRACK_COLOR = "#e3e7ec"  # Light gray track for the confidence meter bar.
CARD_BG = "#f8fafc"      # Soft light background for the About documentation pane.

# ---- Curated Philippine News Headlines for Quick Testing ----
EXAMPLES = [
    "Philippine stocks rally after strong corporate earnings.",
    "Oil prices rise as global demand strengthens.",
    "BSP keeps policy rates unchanged amid steady growth.",
    "Inflation hits new high as rice prices surge across Luzon.",
    "Typhoon leaves thousands without power and forces evacuations.",
    "Audit uncovers widespread corruption in public infrastructure agency.",
    "Commuters stranded as transport strike paralyzes Metro Manila.",
    "Local farmers celebrate record palay harvest after irrigation upgrades.",
    "Peso weakens as foreign investors react to deficit concerns.",
    "Residents welcome the long-awaited flood control project.",
]


def build_about_text(meta: dict | None) -> str:
    """Format clear educational & Responsible-AI documentation from model_metadata.json."""
    m = meta or {}
    model_name = m.get("model_name", "Tuned Logistic Regression")
    train_rows = f"{m['train_rows']:,}" if isinstance(m.get("train_rows"), int) else "N/A"
    test_rows = f"{m['test_rows']:,}" if isinstance(m.get("test_rows"), int) else "N/A"
    date_span = f"{m.get('date_start', '2014-01-01')} to {m.get('date_end', '2023-05-14')}"
    params = m.get("best_params", {})
    return (
        "WHAT THIS APPLICATION DOES\n"
        "This lightweight offline Tkinter desktop application classifies Philippine news headlines as\n"
        "Positive or Negative using the trained Rule-Based NLP pipeline from Rule-Based/news-headline.ipynb.\n\n"
        "PIPELINE ARCHITECTURE\n"
        f"  - Classifier     : {model_name} (calibrated class probabilities)\n"
        "  - Text Features  : TF-IDF word unigrams & bigrams (with negation words preserved)\n"
        "  - Meta Features  : Headline length, word count, punctuation, digit/uppercase ratios,\n"
        "                     and negative domain-term indicator (typhoon, flood, inflation, corruption, etc.)\n"
        f"  - Best Params    : {params}\n"
        f"  - Dataset Split  : {train_rows} train rows / {test_rows} held-out test rows ({date_span})\n\n"
        "LIVE FEATURE CONTRIBUTION TABLE\n"
        "For every headline analyzed, the table lists the exact active TF-IDF words/bigrams and metadata\n"
        "features that contributed most to the Logistic Regression log-odds (Value x Coefficient).\n"
        "  - Positive Contribution (+) : Pushes prediction toward POSITIVE.\n"
        "  - Negative Contribution (-) : Pushes prediction toward NEGATIVE.\n\n"
        "IMPORTANT LIMITATIONS & RESPONSIBLE USE\n"
        "  1. Pseudo-Label Target: Training labels were derived from TextBlob polarity (> +0.05 Positive,\n"
        "     < -0.05 Negative, near-neutral dropped) plus domain overrides — NOT human annotators.\n"
        "  2. Bag-of-Words Limits: Because TF-IDF relies on individual n-grams, it can miss subtle\n"
        "     negation or multi-clause contrast (compare with Semantic/app_tk.py for full-sentence context).\n"
        "  3. English-oriented archive; Filipino/Taglish headlines are not formally validated.\n"
        "  4. High feature weights show model association, NOT causal proof.\n"
        "  5. Analytical & educational aid only — never use for financial trading or high-stakes decisions."
    )


class Meter(tk.Canvas):
    """Horizontal confidence bar widget rendering a fraction cleanly in [0.0, 1.0]."""

    def __init__(self, master, width=420, height=15):
        super().__init__(master, width=width, height=height, highlightthickness=0, bg=TRACK_COLOR)
        self.bar_width = width
        self.bar_height = height
        self._bar = self.create_rectangle(0, 0, 0, height, width=0, fill=MUTED)

    def set(self, fraction: float, color: str):
        clamped = max(0.0, min(1.0, float(fraction)))
        self.coords(self._bar, 0, 0, self.bar_width * clamped, self.bar_height)
        self.itemconfigure(self._bar, fill=color)


class RuleBasedApp(tk.Tk):
    """Main Tkinter desktop application for the Rule-Based (TF-IDF + Logistic Regression) model."""

    def __init__(self):
        super().__init__()
        self.title("PH News Headline Sentiment — Rule-Based (TF-IDF + Logistic Regression)")
        self.geometry("1040x760")
        self.minsize(880, 660)

        self.queue = queue.Queue()
        self.predictor = None
        self.meta = None
        self.busy = False

        self.status = tk.StringVar(value="Loading Rule-Based sentiment model...")
        self.msg = tk.StringVar(value="")

        self._configure_style()
        ttk.Label(self, textvariable=self.status, foreground=MUTED).pack(side="bottom", anchor="w", padx=14, pady=(0, 8))

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)
        self.tab_analyze = ttk.Frame(tabs)
        self.tab_history = ttk.Frame(tabs)
        self.tab_about = ttk.Frame(tabs)
        tabs.add(self.tab_analyze, text="  Analyze  ")
        tabs.add(self.tab_history, text="  History  ")
        tabs.add(self.tab_about, text="  About  ")

        self._build_analyze()
        self._build_history()
        self._build_about()

        self.bind("<Control-Return>", lambda _e: self.analyze())
        self.after(60, self._poll)
        threading.Thread(target=self._load_model, daemon=True).start()

    # ------------------------------------------------------------------ UI Construction
    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", padding=(12, 6), font=("Segoe UI", 10))

    def _build_analyze(self):
        f = self.tab_analyze
        ttk.Label(f, text="Philippine News Headline Sentiment — Rule-Based (TF-IDF + LR)", font=("Segoe UI", 16, "bold")).pack(
            anchor="w", padx=12, pady=(12, 0)
        )
        ttk.Label(
            f,
            text="Fast offline TF-IDF unigram/bigram + headline metadata classifier with live Logistic Regression feature weights.",
            foreground=MUTED,
        ).pack(anchor="w", padx=12, pady=(0, 8))

        # Headline input box
        box = ttk.LabelFrame(f, text="Enter a Philippine News Headline (up to 500 characters)")
        box.pack(fill="x", padx=12, pady=4)
        self.text = tk.Text(box, height=3, wrap="word", font=("Segoe UI", 12), relief="flat", bg="white")
        self.text.pack(fill="x", padx=8, pady=8)
        self.text.focus_set()

        # Action bar: Analyze, Clear, and Example selector
        row = ttk.Frame(f)
        row.pack(fill="x", padx=12, pady=4)
        self.btn = ttk.Button(row, text="Predict Sentiment  (Ctrl+Enter)", command=self.analyze, state="disabled")
        self.btn.pack(side="left")
        ttk.Button(row, text="Clear", command=self.clear).pack(side="left", padx=6)
        self.example = ttk.Combobox(row, values=EXAMPLES, state="readonly", width=58)
        self.example.set("Try a sample Philippine news headline...")
        self.example.pack(side="right")
        self.example.bind("<<ComboboxSelected>>", self._pick_example)

        ttk.Label(f, textvariable=self.msg, foreground=NEG_COLOR, font=("Segoe UI", 10)).pack(anchor="w", padx=14)

        # Prediction summary card
        card = ttk.LabelFrame(f, text="Prediction Result (Rule-Based TF-IDF + Logistic Regression)")
        card.pack(fill="x", padx=12, pady=6)
        self.lbl_pred = ttk.Label(card, text="-", font=("Segoe UI", 24, "bold"), foreground=MUTED)
        self.lbl_pred.pack(anchor="w", padx=14, pady=(8, 0))
        self.lbl_conf = ttk.Label(card, text="", font=("Segoe UI", 10, "bold"), foreground=MUTED)
        self.lbl_conf.pack(anchor="w", padx=14, pady=(2, 0))
        self.lbl_probs = ttk.Label(card, text="", font=("Segoe UI", 9), foreground=MUTED)
        self.lbl_probs.pack(anchor="w", padx=14, pady=(1, 4))
        self.meter = Meter(card, width=480, height=15)
        self.meter.pack(anchor="w", padx=14, pady=(2, 10))

        # Live TF-IDF + Metadata Feature Contributions Table
        tbl_box = ttk.LabelFrame(f, text="Top Active Feature Contributions (TF-IDF Words/Phrases & Metadata)")
        tbl_box.pack(fill="both", expand=True, padx=12, pady=(2, 6))
        cols = ("feature", "kind", "weight", "contrib", "toward")
        self.tree = ttk.Treeview(tbl_box, columns=cols, show="headings", height=7)
        for key, heading, w, anc in (
            ("feature", "Active Feature / Word", 320, "w"),
            ("kind", "Feature Type", 170, "center"),
            ("weight", "LR Weight", 110, "center"),
            ("contrib", "Contribution", 120, "center"),
            ("toward", "Pushes Toward", 130, "center"),
        ):
            self.tree.heading(key, text=heading)
            self.tree.column(key, width=w, anchor=anc)
        self.tree.tag_configure("pos", foreground=POS_COLOR)
        self.tree.tag_configure("neg", foreground=NEG_COLOR)

        scroll = ttk.Scrollbar(tbl_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)

        ttk.Label(
            f,
            foreground=MUTED,
            wraplength=980,
            text="Contribution = Active Feature Value x Logistic Regression Coefficient. "
                 "Positive values pull toward Positive; negative values pull toward Negative. "
                 "(Model association with TextBlob pseudo-labels, not causal proof.)",
        ).pack(anchor="w", padx=14, pady=(0, 8))

    def _build_history(self):
        f = self.tab_history
        wrap = ttk.Frame(f)
        wrap.pack(fill="both", expand=True, padx=12, pady=(12, 6))
        cols = ("time", "headline", "pred", "conf", "p_pos", "p_neg")
        self.history_tree = ttk.Treeview(wrap, columns=cols, show="headings")
        for key, heading, w, anc in (
            ("time", "Time", 80, "center"),
            ("headline", "Headline", 490, "w"),
            ("pred", "Prediction", 100, "center"),
            ("conf", "Confidence", 90, "center"),
            ("p_pos", "P(Pos)", 80, "center"),
            ("p_neg", "P(Neg)", 80, "center"),
        ):
            self.history_tree.heading(key, text=heading)
            self.history_tree.column(key, width=w, anchor=anc)

        h_scroll = ttk.Scrollbar(wrap, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=h_scroll.set)
        self.history_tree.pack(side="left", fill="both", expand=True)
        h_scroll.pack(side="right", fill="y")

        row = ttk.Frame(f)
        row.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(row, text="Export History to CSV", command=self.export_history).pack(side="left")
        ttk.Button(row, text="Clear History", command=self.clear_history).pack(side="left", padx=6)

    def _build_about(self):
        wrap = ttk.Frame(self.tab_about)
        wrap.pack(fill="both", expand=True, padx=14, pady=14)
        self.about_box = tk.Text(wrap, wrap="word", font=("Segoe UI", 10), bg=CARD_BG, relief="flat", padx=14, pady=14)
        scroll = ttk.Scrollbar(wrap, orient="vertical", command=self.about_box.yview)
        self.about_box.configure(yscrollcommand=scroll.set)
        self.about_box.insert("1.0", build_about_text(None))
        self.about_box.configure(state="disabled")
        self.about_box.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    # ------------------------------------------------------------------ Actions & Worker Threading
    def _pick_example(self, _event=None):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.example.get())
        self.msg.set("")

    def clear(self):
        self.text.delete("1.0", "end")
        self.msg.set("")
        self.lbl_pred.configure(text="-", foreground=MUTED)
        self.lbl_conf.configure(text="")
        self.lbl_probs.configure(text="")
        self.meter.set(0.0, MUTED)
        self.tree.delete(*self.tree.get_children())
        self.text.focus_set()

    def analyze(self):
        if self.busy or self.predictor is None:
            return
        headline = self.text.get("1.0", "end").strip()
        if not headline:
            self.msg.set("Please type or select a news headline first.")
            return
        self.msg.set("")
        self.busy = True
        self.btn.configure(state="disabled")
        self.status.set("Running TF-IDF + Logistic Regression prediction...")
        threading.Thread(target=self._work, args=(headline,), daemon=True).start()

    def _work(self, headline: str):
        try:
            t0 = time.perf_counter()
            res = self.predictor.predict_one(headline)
            feats = self.predictor.explain_one(headline, top_k=10)
            ms = (time.perf_counter() - t0) * 1000.0
            self.queue.put(("result", {"res": res, "feats": feats, "ms": ms}))
        except ValueError as exc:
            self.queue.put(("invalid", str(exc)))
        except Exception as exc:
            self.queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _load_model(self):
        try:
            from news_predictor import NewsSentimentPredictor
            art_dir = BASE / "news_sentiment_artifacts"
            model_file = art_dir / "ph_news_sentiment_pipeline.joblib"
            meta_file = art_dir / "model_metadata.json"
            pred = NewsSentimentPredictor(str(model_file))
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            self.queue.put(("loaded", (pred, meta)))
        except Exception as exc:
            self.queue.put((
                "fatal",
                f"Could not load the Rule-Based model:\n{type(exc).__name__}: {exc}\n\n"
                "Make sure you have run Rule-Based/news-headline.ipynb first (which saves "
                "news_sentiment_artifacts/ph_news_sentiment_pipeline.joblib).",
            ))

    def _poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "loaded":
                    self.predictor, self.meta = payload
                    self.btn.configure(state="normal")
                    mname = self.meta.get("model_name", "Tuned Logistic Regression")
                    mver = self.meta.get("model_version", "1.0")
                    self.status.set(f"Ready — Loaded Rule-Based {mname} (v{mver})")
                    self.about_box.configure(state="normal")
                    self.about_box.delete("1.0", "end")
                    self.about_box.insert("1.0", build_about_text(self.meta))
                    self.about_box.configure(state="disabled")
                elif kind == "fatal":
                    self.status.set("Model load failed.")
                    messagebox.showerror("Model Load Error", payload)
                elif kind == "result":
                    self._render_result(payload)
                    self.busy = False
                    self.btn.configure(state="normal")
                    self.status.set(f"Done ({payload['ms']:.1f} ms) — Ready")
                elif kind == "invalid":
                    self.msg.set(payload)
                    self.busy = False
                    self.btn.configure(state="normal")
                    self.status.set("Ready")
                elif kind == "error":
                    self.busy = False
                    self.btn.configure(state="normal")
                    self.status.set("Error during prediction.")
                    messagebox.showerror("Prediction Error", payload)
        except queue.Empty:
            pass
        self.after(60, self._poll)

    def _render_result(self, payload: dict):
        res, feats = payload["res"], payload["feats"]
        label = res["prediction"]
        conf = res["confidence"]
        p_pos = res["probabilities"].get("Positive", 0.0)
        p_neg = res["probabilities"].get("Negative", 0.0)
        color = POS_COLOR if label == "Positive" else NEG_COLOR

        self.lbl_pred.configure(text=label.upper(), foreground=color)
        self.lbl_conf.configure(text=f"{conf:.1%} confident  ({payload['ms']:.1f} ms)", foreground="#1e293b")
        self.lbl_probs.configure(text=f"P(Positive): {p_pos:.1%}   |   P(Negative): {p_neg:.1%}", foreground=MUTED)
        self.meter.set(conf, color)

        self.tree.delete(*self.tree.get_children())
        for it in feats:
            tag = "pos" if it["pushes_toward"] == "Positive" else "neg"
            self.tree.insert(
                "", "end",
                values=(it["feature"], it["kind"], f"{it['weight']:+.3f}", f"{it['contribution']:+.3f}", it["pushes_toward"]),
                tags=(tag,),
            )

        now_str = dt.datetime.now().strftime("%H:%M:%S")
        self.history_tree.insert(
            "", 0,
            values=(now_str, res["headline"], label, f"{conf:.1%}", f"{p_pos:.1%}", f"{p_neg:.1%}"),
        )

    def export_history(self):
        rows = [self.history_tree.item(i, "values") for i in self.history_tree.get_children()]
        if not rows:
            messagebox.showinfo("Export History", "No predictions in history yet.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="rule_based_sentiment_history.csv",
            title="Export Prediction History",
        )
        if not save_path:
            return
        with open(save_path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["Time", "Headline", "Prediction", "Confidence", "P(Positive)", "P(Negative)"])
            w.writerows(rows)
        self.status.set(f"Exported {len(rows)} record(s) to {Path(save_path).name}")

    def clear_history(self):
        self.history_tree.delete(*self.history_tree.get_children())
        self.status.set("History cleared.")


if __name__ == "__main__":
    RuleBasedApp().mainloop()
