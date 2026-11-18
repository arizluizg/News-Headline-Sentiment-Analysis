"""app_tk.py - desktop (tkinter) interface for the semantic, context-aware headline sentiment model.

Run it from the folder that contains semantic_predictor.py and the semantic_artifacts/ folder
(both are created by news-headline-semantic.ipynb):

    python app_tk.py

If the rule-based model from the first notebook is also in this folder (news_predictor.py,
news_nlp_utils.py and news_sentiment_artifacts/), the app shows both predictions side by side.
"""
import csv
import datetime as dt
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

POS_COLOR = "#1a9e4b"
NEG_COLOR = "#d64545"
WARN_COLOR = "#b7791f"
MUTED = "#5f6b7a"
TRACK_COLOR = "#e3e7ec"

EXAMPLES = [
    "The president is not bad",
    "Inflation is no longer a problem for families",
    "No corruption found in the audit",
    "Officials fail to impress investors, but residents welcome the new bridge",
    "Typhoon leaves thousands without power while rescuers work through the night",
    "Rice prices hit a new high",
    "Unemployment falls to a record low",
]

ABOUT_TEXT = (
    "What this app does\n"
    "It classifies a news headline as Positive or Negative by reading the whole sentence, not just single words. "
    "A sentence-embedding model turns the headline into a vector of meaning (so 'not bad' lands near 'good'), "
    "and a trained classifier turns that vector into a prediction.\n\n"
    "Phrase-by-phrase breakdown\n"
    "The headline is split into phrases (at commas or words like 'but' / 'while'; otherwise 3-word chunks).\n"
    "  - Alone: how positive the phrase reads by itself.\n"
    "  - Impact: P(Positive) of the full headline minus P(Positive) with that phrase removed. "
    "A positive number means the phrase pushes the headline toward Positive; negative means toward Negative.\n\n"
    "Rule-based comparison\n"
    "When the first notebook's model is available, its prediction is shown next to the semantic one. "
    "It scores words individually, so negation and contrast are often missed.\n\n"
    "Limitations\n"
    "  - Training labels come from a pretrained sentiment model, not human annotators.\n"
    "  - English only; Filipino / Taglish headlines are untested.\n"
    "  - The explanation is heuristic (association, not proof of cause).\n"
    "  - An automated aid only - not for financial or other high-stakes decisions."
)


class Meter(tk.Canvas):
    """A small coloured bar showing a value between 0 and 1."""

    def __init__(self, master, width=300, height=14):
        super().__init__(master, width=width, height=height, highlightthickness=0, bg=TRACK_COLOR)
        self.bar_width, self.bar_height = width, height
        self._bar = self.create_rectangle(0, 0, 0, height, width=0, fill=MUTED)

    def set(self, fraction, color):
        fraction = max(0.0, min(1.0, float(fraction)))
        self.coords(self._bar, 0, 0, self.bar_width * fraction, self.bar_height)
        self.itemconfigure(self._bar, fill=color)
        
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PH News Headline Sentiment - semantic (context-aware)")
        self.geometry("1020x780")
        self.minsize(920, 700)

        self.queue = queue.Queue()   # worker threads -> UI thread (tkinter is not thread-safe)
        self.semantic = None
        self.rule = None
        self.busy = False

        self.status = tk.StringVar(value="Loading models... (the first start can take 10-30 seconds)")
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

        self.bind("<Control-Return>", lambda _event: self.analyze())
        self.after(100, self._poll)
        threading.Thread(target=self._load_models, daemon=True).start()

    # ------------------------------------------------------------------ layout
    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", padding=(12, 6))

    def _card(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title)
        pred = ttk.Label(frame, text="-", font=("Segoe UI", 22, "bold"), foreground=MUTED)
        pred.pack(anchor="w", padx=12, pady=(8, 0))
        conf = ttk.Label(frame, text="", foreground=MUTED)
        conf.pack(anchor="w", padx=12)
        meter = Meter(frame, width=320, height=14)
        meter.pack(anchor="w", padx=12, pady=(4, 12))
        return {"frame": frame, "pred": pred, "conf": conf, "meter": meter}

    def _build_analyze(self):
        f = self.tab_analyze
        ttk.Label(f, text="Headline sentiment - semantic (context-aware)", font=("Segoe UI", 16, "bold")).pack(
            anchor="w", padx=12, pady=(12, 0))
        ttk.Label(f, text="Reads the whole sentence.",
                  foreground=MUTED).pack(anchor="w", padx=12, pady=(0, 8))

        box = ttk.LabelFrame(f, text="Headline")
        box.pack(fill="x", padx=12, pady=4)
        self.text = tk.Text(box, height=4, wrap="word", font=("Segoe UI", 12), relief="flat")
        self.text.pack(fill="x", padx=8, pady=8)
        self.text.focus_set()

        row = ttk.Frame(f)
        row.pack(fill="x", padx=12, pady=4)
        self.btn = ttk.Button(row, text="Analyze  (Ctrl+Enter)", command=self.analyze, state="disabled")
        self.btn.pack(side="left")
        ttk.Button(row, text="Clear", command=self.clear).pack(side="left", padx=6)
        self.example = ttk.Combobox(row, values=EXAMPLES, state="readonly", width=58)
        self.example.set("Try an example...")
        self.example.pack(side="right")
        self.example.bind("<<ComboboxSelected>>", self._pick_example)

        ttk.Label(f, textvariable=self.msg, foreground=NEG_COLOR).pack(anchor="w", padx=14)

        cards = ttk.Frame(f)
        cards.pack(fill="x", padx=12, pady=6)
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)
        self.card_sem = self._card(cards, "Semantic model (context-aware)")
        self.card_rule = self._card(cards, "Rule-based model (word-by-word)")
        self.card_sem["frame"].grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.card_rule["frame"].grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self.agree = ttk.Label(f, text="", font=("Segoe UI", 10, "bold"))
        self.agree.pack(anchor="w", padx=14, pady=(0, 4))

        table = ttk.LabelFrame(f, text="Phrase-by-phrase breakdown (semantic model)")
        table.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        columns = ("phrase", "alone", "impact", "toward")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", height=6)
        for name, label, width, anchor in (("phrase", "Phrase", 470, "w"), ("alone", "Alone: P(Positive)", 130, "center"),
                                           ("impact", "Impact", 90, "center"), ("toward", "Pushes toward", 130, "center")):
            self.tree.heading(name, text=label)
            self.tree.column(name, width=width, anchor=anchor)
        self.tree.tag_configure("pos", foreground=POS_COLOR)
        self.tree.tag_configure("neg", foreground=NEG_COLOR)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)

        ttk.Label(f, foreground=MUTED, wraplength=940,
                  text="Impact = P(Positive) of the full headline minus P(Positive) with that phrase removed. "
                       "Positive impact pushes the headline toward Positive, negative toward Negative.").pack(
            anchor="w", padx=14, pady=(0, 8))

    def _build_history(self):
        f = self.tab_history
        columns = ("time", "headline", "semantic", "sem_conf", "rule", "rule_conf")
        self.history_tree = ttk.Treeview(f, columns=columns, show="headings")
        for name, label, width, anchor in (("time", "Time", 80, "center"), ("headline", "Headline", 460, "w"),
                                           ("semantic", "Semantic", 90, "center"), ("sem_conf", "Conf.", 70, "center"),
                                           ("rule", "Rule-based", 90, "center"), ("rule_conf", "Conf.", 70, "center")):
            self.history_tree.heading(name, text=label)
            self.history_tree.column(name, width=width, anchor=anchor)
        self.history_tree.pack(fill="both", expand=True, padx=12, pady=(12, 6))
        row = ttk.Frame(f)
        row.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(row, text="Export to CSV", command=self.export_history).pack(side="left")
        ttk.Button(row, text="Clear history", command=self.clear_history).pack(side="left", padx=6)

    def _build_about(self):
        label = ttk.Label(self.tab_about, text=ABOUT_TEXT, justify="left", wraplength=900)
        label.pack(anchor="nw", padx=18, pady=18)

    # ------------------------------------------------------------------ actions
    def _pick_example(self, _event=None):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.example.get())

    def clear(self):
        self.text.delete("1.0", "end")
        self.msg.set("")
        self.agree.configure(text="")
        self.tree.delete(*self.tree.get_children())
        for card in (self.card_sem, self.card_rule):
            self._reset_card(card)
        self.text.focus_set()

    def analyze(self):
        if self.busy or self.semantic is None:
            return
        headline = self.text.get("1.0", "end").strip()
        if not headline:
            self.msg.set("Please type a headline first.")
            return
        self.msg.set("")
        self.busy = True
        self.btn.configure(state="disabled")
        self.status.set("Analyzing...")
        threading.Thread(target=self._work, args=(headline,), daemon=True).start()

    def _work(self, headline):
        """Runs in a worker thread: results go back to the UI thread through the queue."""
        try:
            semantic = self.semantic.predict_one(headline)
            phrases = self.semantic.explain_one(headline)
            rule = None
            if self.rule is not None:
                try:
                    rule = self.rule.predict_one(headline)
                except Exception:
                    rule = None
            self.queue.put(("result", {"sem": semantic, "phrases": phrases, "rule": rule}))
        except ValueError as exc:
            self.queue.put(("invalid", str(exc)))
        except Exception as exc:
            self.queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _load_models(self):
        try:
            from semantic_predictor import SemanticSentimentPredictor
            semantic = SemanticSentimentPredictor(str(BASE / "semantic_artifacts"))
        except Exception as exc:
            self.queue.put(("fatal", f"Could not load the semantic model:\n{type(exc).__name__}: {exc}\n\n"
                                     "Run news-headline-semantic.ipynb first (it creates semantic_artifacts/ and "
                                     "semantic_predictor.py) and make sure torch, transformers and "
                                     "sentence-transformers are installed."))
            return
        rule = None
        artifact = BASE / "news_sentiment_artifacts" / "ph_news_sentiment_pipeline.joblib"
        if artifact.exists():
            try:
                from news_predictor import NewsSentimentPredictor
                rule = NewsSentimentPredictor(str(artifact))
            except Exception:
                rule = None
        self.queue.put(("ready", (semantic, rule)))

    def _poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "ready":
                    self.semantic, self.rule = payload
                    self.btn.configure(state="normal")
                    note = "" if self.rule is not None else "  (rule-based model not found - showing semantic only)"
                    self.status.set("Ready." + note)
                    if self.rule is None:
                        self._reset_card(self.card_rule, "not loaded")
                elif kind == "fatal":
                    self.status.set("Could not load the model.")
                    messagebox.showerror("Model not available", payload)
                elif kind == "result":
                    self._show(payload)
                    self._finish("Done.")
                elif kind == "invalid":
                    self.msg.set(payload)
                    self._finish("Ready.")
                elif kind == "error":
                    self.msg.set("Something went wrong: " + payload)
                    self._finish("Ready.")
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _finish(self, status):
        self.busy = False
        self.btn.configure(state="normal")
        self.status.set(status)

    # ------------------------------------------------------------------ display
    @staticmethod
    def _color(label):
        return POS_COLOR if label == "Positive" else NEG_COLOR

    def _reset_card(self, card, text="-"):
        card["pred"].configure(text=text, foreground=MUTED)
        card["conf"].configure(text="")
        card["meter"].set(0, MUTED)

    def _fill_card(self, card, result):
        color = self._color(result["prediction"])
        card["pred"].configure(text=result["prediction"].upper(), foreground=color)
        card["conf"].configure(text=f"{result['confidence']:.0%} confident")
        card["meter"].set(result["confidence"], color)

    def _show(self, data):
        sem, phrases, rule = data["sem"], data["phrases"], data["rule"]
        self._fill_card(self.card_sem, sem)
        if rule is not None:
            self._fill_card(self.card_rule, rule)
            if rule["prediction"] == sem["prediction"]:
                self.agree.configure(text="Both approaches agree.", foreground=MUTED)
            else:
                self.agree.configure(text="The two approaches disagree - check the phrase breakdown to see why.",
                                     foreground=WARN_COLOR)
        else:
            self._reset_card(self.card_rule, "not loaded")
            self.agree.configure(text="")

        self.tree.delete(*self.tree.get_children())
        for item in sorted(phrases, key=lambda d: d["order"]):
            tag = "pos" if item["pushes_toward"] == "Positive" else "neg"
            self.tree.insert("", "end", tags=(tag,), values=(
                item["phrase"], f"{item['alone_pos']:.0%}", f"{item['impact']:+.2f}", item["pushes_toward"]))

        self.history_tree.insert("", 0, values=(
            dt.datetime.now().strftime("%H:%M:%S"), sem["headline"], sem["prediction"], f"{sem['confidence']:.0%}",
            rule["prediction"] if rule else "-", f"{rule['confidence']:.0%}" if rule else "-"))

    # ------------------------------------------------------------------ history
    def export_history(self):
        rows = [self.history_tree.item(i, "values") for i in self.history_tree.get_children()]
        if not rows:
            messagebox.showinfo("Nothing to export", "No predictions yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
                                            initialfile="sentiment_history.csv")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "headline", "semantic", "semantic_confidence", "rule_based", "rule_confidence"])
            writer.writerows(rows)
        self.status.set(f"Exported {len(rows)} rows to {path}")

    def clear_history(self):
        self.history_tree.delete(*self.history_tree.get_children())


if __name__ == "__main__":
    try:  # sharper text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()
