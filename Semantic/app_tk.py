"""app_tk.py — Tkinter desktop application for Philippines News Headlines Semantic Sentiment Analysis.

Run this application from the Semantic/ directory after running news-headline-semantic.ipynb
(which produces semantic_predictor.py and the offline model folder semantic_artifacts/):

    python app_tk.py

If the companion rule-based model exists either locally (Semantic/news_sentiment_artifacts/)
or in the sibling folder (../Rule-Based/news_sentiment_artifacts/), the interface automatically
loads both models and shows their predictions side by side.
"""

import csv  # Imports csv so the user can export their prediction history table to a spreadsheet file.
import datetime as dt  # Imports datetime to timestamp every analyzed headline in the History tab.
import queue  # Imports thread-safe Queue so background inference threads can safely pass results to the Tkinter UI thread.
import sys  # Imports sys so we can add local and sibling Rule-Based directories to Python's module search path.
import threading  # Imports threading so model loading and inference run in background threads without freezing the GUI.
import tkinter as tk  # Imports Tkinter, Python's built-in desktop GUI toolkit.
from pathlib import Path  # Imports Path for clean, OS-independent resolution of model artifact folders.
from tkinter import filedialog, messagebox, ttk  # Imports themed Tkinter widgets, file save dialogs, and error popups.

BASE = Path(__file__).resolve().parent  # Resolves the absolute path of the Semantic/ folder containing app_tk.py.
if str(BASE) not in sys.path:  # Ensures Python can import semantic_predictor.py regardless of working directory.
    sys.path.insert(0, str(BASE))  # Prepends Semantic/ to sys.path cleanly.

# ---- Visual Design Palette (accessible high-contrast colors for sentiment distinction) ----
POS_COLOR = "#1a9e4b"  # Green color representing Positive sentiment predictions and positive phrase impacts.
NEG_COLOR = "#d64545"  # Red color representing Negative sentiment predictions and negative phrase impacts.
WARN_COLOR = "#b7791f"  # Amber color highlighting disagreement between the semantic and rule-based models.
MUTED = "#5f6b7a"  # Slate gray used for secondary labels, hints, and status bar messages.
TRACK_COLOR = "#e3e7ec"  # Soft neutral track background for the visual confidence meter bar.
CARD_BG = "#f8fafc"  # Subtle light neutral tone for read-only information text panes.

# ---- Curated Examples Showcasing Negation, Contrast, and Philippine News Context ----
EXAMPLES = [
    "The president is not bad, but the economy is struggling.",
    "Despite the crisis, economic growth remains strong.",
    "Residents welcome the project, although critics raise concerns.",
    "Inflation is no longer a problem for families.",
    "No corruption found in the audit.",
    "The new policy is good, but its implementation is poor.",
    "The president is not bad.",
    "The government is not doing well.",
    "Officials fail to impress investors, but residents welcome the new bridge.",
    "Typhoon leaves thousands without power while rescuers work through the night.",
    "Not only good but excellent.",
    "Rice prices hit a new high.",
    "The Mayor donated thousands to the charity, but the barangay officials stole the funds.",
]

# ---- Educational Help & Responsible-AI Documentation Shown in the About Tab ----
ABOUT_TEXT = (
    "WHAT THIS APPLICATION DOES\n"
    "This desktop application classifies Philippine news headlines as Positive or Negative by reading the\n"
    "complete sentence context rather than scoring isolated words. A partially fine-tuned pretrained\n"
    "Transformer encoder (distilbert-base-uncased, top 2 of 6 layers adapted on all 64,341 headlines)\n"
    "extracts CLS + Mean contextual sentence vectors, and a PyTorch Multi-Layer Perceptron (MLP) head\n"
    "with Batch Normalization, ReLU, and Dropout computes Positive and Negative probabilities.\n\n"
    "PHRASE-BY-PHRASE SEMANTIC BREAKDOWN\n"
    "Each headline is segmented along meaningful linguistic clause boundaries (commas, semicolons,\n"
    "'but', 'although', 'despite', 'while', 'whereas') while preserving negation and multi-word idioms\n"
    "such as 'not bad', 'no longer', 'failed to deliver', and 'despite the crisis'.\n"
    "  - Phrase Sentiment : How the model classifies that phrase when read completely on its own.\n"
    "  - Alone P(Pos) / P(Neg) : Standalone Positive and Negative probabilities of the extracted phrase.\n"
    "  - Contribution (Impact) : Full-headline P(Positive) minus P(Positive) with that phrase removed.\n"
    "    A positive contribution pulls the full headline toward Positive; a negative contribution pulls\n"
    "    it toward Negative.\n\n"
    "COMPANION RULE-BASED COMPARISON\n"
    "When the companion TF-IDF + Logistic Regression model from Rule-Based/news_sentiment_artifacts/ is\n"
    "found, its prediction is shown alongside the semantic model. Because bag-of-words approaches rely\n"
    "heavily on individual word counts, they frequently struggle with negation and contrast clauses.\n\n"
    "IMPORTANT LIMITATIONS & RESPONSIBLE USE\n"
    "  1. Training labels were generated by a pretrained Transformer teacher (DistilBERT SST-2), not\n"
    "     human annotators; test metrics measure agreement with teacher labels unless human validation\n"
    "     is completed in Section 14 of news-headline-semantic.ipynb.\n"
    "  2. Binary Positive / Negative output only (no separate Neutral class for purely factual headlines).\n"
    "  3. English-oriented models; Filipino, Taglish, and regional languages are not formally validated.\n"
    "  4. Phrase occlusion impact is a heuristic explanation of model behavior, not causal proof.\n"
    "  5. Educational and analytical aid only — never use for financial trading or high-stakes decisions."
)


class Meter(tk.Canvas):
    """Reusable horizontal progress bar widget displaying a model confidence fraction between 0.0 and 1.0."""

    def __init__(self, master, width=340, height=14):
        """Initialize the empty confidence bar canvas with a neutral gray track."""
        super().__init__(master, width=width, height=height, highlightthickness=0, bg=TRACK_COLOR)  # Creates borderless canvas track.
        self.bar_width = width  # Stores total pixel width representing 100% confidence.
        self.bar_height = height  # Stores bar thickness in pixels.
        self._bar = self.create_rectangle(0, 0, 0, height, width=0, fill=MUTED)  # Draws initial zero-width fill rectangle.

    def set(self, fraction: float, color: str):
        """Update the filled width and color of the confidence meter safely in [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(fraction)))  # Clamps value safely to [0.0, 1.0] to prevent drawing overflow.
        self.coords(self._bar, 0, 0, self.bar_width * clamped, self.bar_height)  # Resizes bar proportionally to confidence.
        self.itemconfigure(self._bar, fill=color)  # Updates bar fill color (green for Positive, red for Negative).


class App(tk.Tk):
    """Main Tkinter desktop window orchestrating background model loading, prediction, phrase breakdown, and history."""

    def __init__(self):
        """Build the tabbed desktop UI, register keyboard shortcuts, and launch asynchronous model loading."""
        super().__init__()  # Initializes the root Tkinter window.
        self.title("PH News Headline Sentiment — Semantic (Context-Aware Deep Learning)")  # Sets descriptive window title.
        self.geometry("1120x820")  # Sets a generous default window size so all phrase columns fit cleanly without clipping.
        self.minsize(960, 720)  # Sets sensible minimum window dimensions so widgets stay readable when resized.

        self.queue = queue.Queue()  # Thread-safe mailbox transferring worker-thread results back to Tkinter's main UI loop.
        self.semantic = None  # Will hold the loaded SemanticSentimentPredictor instance once background load finishes.
        self.rule = None  # Will hold the optional companion NewsSentimentPredictor instance when available.
        self.busy = False  # Guard flag preventing overlapping inference threads if the user clicks Analyze repeatedly.

        self.status = tk.StringVar(value="Loading deep learning models... (first load may take a few seconds)")  # Bottom status text.
        self.msg = tk.StringVar(value="")  # Inline validation/error banner text shown directly beneath action buttons.

        self._configure_style()  # Configures modern ttk widget fonts, row heights, and padding.
        ttk.Label(self, textvariable=self.status, foreground=MUTED).pack(  # Places persistent status bar at window bottom.
            side="bottom", anchor="w", padx=14, pady=(0, 8)
        )

        # Create the three main application tabs: Analyze, History, and About
        tabs = ttk.Notebook(self)  # Creates tabbed notebook container.
        tabs.pack(fill="both", expand=True, padx=10, pady=10)  # Expands tabs to fill main window area.
        self.tab_analyze = ttk.Frame(tabs)  # Tab 1: Headline input, prediction cards, and phrase breakdown table.
        self.tab_history = ttk.Frame(tabs)  # Tab 2: Session prediction log with CSV export.
        self.tab_about = ttk.Frame(tabs)  # Tab 3: Methodology, explanation guide, and responsible-AI caveats.
        tabs.add(self.tab_analyze, text="  Analyze  ")  # Registers Analyze tab.
        tabs.add(self.tab_history, text="  History  ")  # Registers History tab.
        tabs.add(self.tab_about, text="  About  ")  # Registers About tab.

        self._build_analyze()  # Constructs widgets inside the Analyze tab.
        self._build_history()  # Constructs widgets inside the History tab.
        self._build_about()  # Constructs scrollable documentation inside the About tab.

        self.bind("<Control-Return>", lambda _event: self.analyze())  # Enables Ctrl+Enter keyboard shortcut for quick analysis.
        self.after(100, self._poll)  # Schedules periodic 100 ms queue polling on the Tkinter main thread (thread-safe!).
        threading.Thread(target=self._load_models, daemon=True).start()  # Loads heavy Transformer models off the UI thread.

    # ------------------------------------------------------------------ UI Styling & Construction
    def _configure_style(self):
        """Apply a clean, readable ttk theme and typography for Windows/cross-platform desktop displays."""
        style = ttk.Style(self)  # Retrieves ttk styling controller for this window.
        try:  # Applies the modern 'clam' theme when supported by the OS Tk installation.
            style.theme_use("clam")  # Uses crisp, platform-consistent flat ttk styling.
        except tk.TclError:  # Falls back gracefully to the default OS theme if 'clam' is unavailable.
            pass  # Keeps native theme without raising an error.
        style.configure("Treeview", rowheight=27, font=("Segoe UI", 10))  # Gives table rows comfortable spacing and readable font.
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))  # Makes table headers stand out clearly.
        style.configure("TButton", padding=(12, 6), font=("Segoe UI", 10))  # Adds clean button padding.

    def _card(self, parent, title: str) -> dict:
        """Create a standardized sentiment result card displaying class label, confidence, P(Pos)/P(Neg), and meter."""
        frame = ttk.LabelFrame(parent, text=title)  # Creates titled outer border box for the model card.
        pred = ttk.Label(frame, text="-", font=("Segoe UI", 22, "bold"), foreground=MUTED)  # Prominent Positive/Negative badge.
        pred.pack(anchor="w", padx=12, pady=(8, 0))  # Packs sentiment heading cleanly on the left.
        conf = ttk.Label(frame, text="", font=("Segoe UI", 10, "bold"), foreground=MUTED)  # Displays overall confidence percentage.
        conf.pack(anchor="w", padx=12, pady=(2, 0))  # Packs confidence line right below prediction.
        probs = ttk.Label(frame, text="", font=("Segoe UI", 9), foreground=MUTED)  # Displays explicit P(Positive) and P(Negative).
        probs.pack(anchor="w", padx=12, pady=(1, 2))  # Packs probability breakdown cleanly.
        meter = Meter(frame, width=360, height=14)  # Visual horizontal bar scaled to model confidence.
        meter.pack(anchor="w", padx=12, pady=(4, 10))  # Packs meter at bottom of card.
        return {"frame": frame, "pred": pred, "conf": conf, "probs": probs, "meter": meter}  # Returns widget handles dict.

    def _build_analyze(self):
        """Construct all input controls, side-by-side model cards, and the phrase-by-phrase table in Tab 1."""
        f = self.tab_analyze  # Shorthand reference to the Analyze tab frame.
        ttk.Label(f, text="Philippine News Headline Sentiment — Semantic (Context-Aware)", font=("Segoe UI", 16, "bold")).pack(
            anchor="w", padx=12, pady=(12, 0)  # Renders main application header.
        )
        self.subtitle_lbl = ttk.Label(  # Displays concise explanation of the active model pipeline.
            f,
            text="Evaluates complete headline context (negation, contrast, and clause structure) plus phrase-level contribution.",
            foreground=MUTED,
        )
        self.subtitle_lbl.pack(anchor="w", padx=12, pady=(0, 8))  # Packs subtitle cleanly below header.

        # Headline input text box
        box = ttk.LabelFrame(f, text="Enter a Philippine News Headline (up to 300 characters)")  # Labeled input frame.
        box.pack(fill="x", padx=12, pady=4)  # Stretches input box horizontally across window.
        self.text = tk.Text(box, height=3, wrap="word", font=("Segoe UI", 12), relief="flat", bg="white")  # Multi-line input widget.
        self.text.pack(fill="x", padx=8, pady=8)  # Adds clean inner padding around input area.
        self.text.focus_set()  # Places cursor inside headline box immediately on launch.

        # Control bar: Analyze button, Clear button, and curated Examples dropdown
        row = ttk.Frame(f)  # Horizontal bar holding action buttons and example selector.
        row.pack(fill="x", padx=12, pady=4)  # Packs button row neatly.
        self.btn = ttk.Button(row, text="Analyze  (Ctrl+Enter)", command=self.analyze, state="disabled")  # Disabled until model loads.
        self.btn.pack(side="left")  # Aligns primary action button on left.
        ttk.Button(row, text="Clear", command=self.clear).pack(side="left", padx=6)  # Resets text box and results cleanly.
        self.example = ttk.Combobox(row, values=EXAMPLES, state="readonly", width=62)  # Read-only dropdown of stress/demo examples.
        self.example.set("Try a negation or contrast example...")  # Sets inviting placeholder prompt.
        self.example.pack(side="right")  # Aligns dropdown cleanly on right.
        self.example.bind("<<ComboboxSelected>>", self._pick_example)  # Populates headline box when user selects an example.

        ttk.Label(f, textvariable=self.msg, foreground=NEG_COLOR, font=("Segoe UI", 10)).pack(anchor="w", padx=14)  # Validation message row.

        # Side-by-side prediction cards (Semantic Deep Learning vs Companion Rule-Based)
        cards = ttk.Frame(f)  # Container holding the two prediction cards in equal columns.
        cards.pack(fill="x", padx=12, pady=6)  # Stretches card container horizontally.
        cards.columnconfigure(0, weight=1)  # Gives Semantic card 50% width.
        cards.columnconfigure(1, weight=1)  # Gives Rule-Based comparison card 50% width.
        self.card_sem = self._card(cards, "Semantic Model (Context-Aware Transformer + MLP)")  # Primary deep learning card.
        self.card_rule = self._card(cards, "Rule-Based Companion (Word-by-Word TF-IDF)")  # Secondary baseline comparison card.
        self.card_sem["frame"].grid(row=0, column=0, sticky="nsew", padx=(0, 6))  # Places Semantic card on left.
        self.card_rule["frame"].grid(row=0, column=1, sticky="nsew", padx=(6, 0))  # Places Rule-Based card on right.

        self.agree = ttk.Label(f, text="", font=("Segoe UI", 10, "bold"))  # Displays whether both models agree or disagree.
        self.agree.pack(anchor="w", padx=14, pady=(0, 4))  # Packs agreement indicator line.

        # Upgraded Phrase-by-Phrase Semantic Breakdown Table (Task 11C & 11F)
        table = ttk.LabelFrame(f, text="Phrase-by-Phrase Semantic Breakdown (Context-Aware Model)")  # Framed breakdown section.
        table.pack(fill="both", expand=True, padx=12, pady=(0, 6))  # Expands table vertically when window grows.
        columns = ("phrase", "sentiment", "p_pos", "p_neg", "impact", "toward")  # Defines all 6 required breakdown columns!
        self.tree = ttk.Treeview(table, columns=columns, show="headings", height=6)  # Builds scrollable Treeview table.
        col_specs = (  # Column keys, display headings, pixel widths, and text alignments.
            ("phrase", "Extracted Phrase", 410, "w"),
            ("sentiment", "Phrase Sentiment", 125, "center"),
            ("p_pos", "Alone P(Pos)", 95, "center"),
            ("p_neg", "Alone P(Neg)", 95, "center"),
            ("impact", "Contribution", 105, "center"),
            ("toward", "Pushes Toward", 125, "center"),
        )
        for col_key, heading_txt, col_w, col_anchor in col_specs:  # Configures every table column cleanly.
            self.tree.heading(col_key, text=heading_txt)  # Sets column header label.
            self.tree.column(col_key, width=col_w, anchor=col_anchor)  # Sets column width and alignment.
        self.tree.tag_configure("pos", foreground=POS_COLOR)  # Colors Positive-pushing phrases green.
        self.tree.tag_configure("neg", foreground=NEG_COLOR)  # Colors Negative-pushing phrases red.

        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)  # Vertical scrollbar for long headlines.
        self.tree.configure(yscrollcommand=scroll.set)  # Connects Treeview scrolling to scrollbar thumb.
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)  # Packs table cleanly.
        scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)  # Places scrollbar along right edge.

        ttk.Label(  # Concise educational footnote clarifying heuristic occlusion contribution vs standalone sentiment.
            f,
            foreground=MUTED,
            wraplength=1060,
            text="Contribution (Impact) = Full-headline P(Positive) minus P(Positive) with that phrase removed. "
                 "Positive values pull the overall prediction toward Positive; negative values pull toward Negative. "
                 "Note: phrase removal is a heuristic explanation of model sensitivity, not causal proof.",
        ).pack(anchor="w", padx=14, pady=(0, 8))

    def _build_history(self):
        """Construct the scrollable prediction history table and CSV export controls in Tab 2."""
        f = self.tab_history  # Shorthand reference to History tab frame.
        hist_frame = ttk.Frame(f)  # Wrapper frame pairing the Treeview with its vertical scrollbar.
        hist_frame.pack(fill="both", expand=True, padx=12, pady=(12, 6))  # Expands cleanly across History tab.

        columns = ("time", "headline", "semantic", "sem_conf", "sem_ppos", "rule", "rule_conf")  # History table columns.
        self.history_tree = ttk.Treeview(hist_frame, columns=columns, show="headings")  # History Treeview widget.
        hist_specs = (  # Column metadata for session prediction log.
            ("time", "Time", 80, "center"),
            ("headline", "Headline", 440, "w"),
            ("semantic", "Semantic", 90, "center"),
            ("sem_conf", "Conf.", 70, "center"),
            ("sem_ppos", "P(Pos)", 70, "center"),
            ("rule", "Rule-Based", 95, "center"),
            ("rule_conf", "Conf.", 70, "center"),
        )
        for col_key, heading_txt, col_w, col_anchor in hist_specs:  # Applies column headers and alignments.
            self.history_tree.heading(col_key, text=heading_txt)  # Sets column heading text.
            self.history_tree.column(col_key, width=col_w, anchor=col_anchor)  # Sets column width and alignment.

        h_scroll = ttk.Scrollbar(hist_frame, orient="vertical", command=self.history_tree.yview)  # Fixes missing scrollbar bug!
        self.history_tree.configure(yscrollcommand=h_scroll.set)  # Connects vertical scrollbar to history list.
        self.history_tree.pack(side="left", fill="both", expand=True)  # Packs history table cleanly.
        h_scroll.pack(side="right", fill="y")  # Docks scrollbar cleanly on the right edge.

        row = ttk.Frame(f)  # Bottom button strip for exporting or clearing prediction history.
        row.pack(fill="x", padx=12, pady=(0, 12))  # Packs action strip cleanly.
        ttk.Button(row, text="Export History to CSV", command=self.export_history).pack(side="left")  # CSV export trigger.
        ttk.Button(row, text="Clear History", command=self.clear_history).pack(side="left", padx=6)  # Clears table rows.

    def _build_about(self):
        """Construct a clean scrollable read-only documentation viewer inside the About tab."""
        container = ttk.Frame(self.tab_about)  # Wrapper pairing text area and scrollbar cleanly.
        container.pack(fill="both", expand=True, padx=14, pady=14)  # Adds clean window margins.
        about_box = tk.Text(  # Scrollable read-only text box prevents clipping on smaller screens.
            container, wrap="word", font=("Segoe UI", 10), bg=CARD_BG, relief="flat", padx=14, pady=14
        )
        about_scroll = ttk.Scrollbar(container, orient="vertical", command=about_box.yview)  # Vertical scrollbar for About text.
        about_box.configure(yscrollcommand=about_scroll.set)  # Connects scrollbar to text widget.
        about_box.insert("1.0", ABOUT_TEXT)  # Inserts educational overview and limitations text.
        about_box.configure(state="disabled")  # Locks widget read-only so users can select/copy text without editing.
        about_box.pack(side="left", fill="both", expand=True)  # Packs text area cleanly.
        about_scroll.pack(side="right", fill="y")  # Packs scrollbar along right edge.

    # ------------------------------------------------------------------ User Actions & Threading
    def _pick_example(self, _event=None):
        """Copy the chosen dropdown example into the headline text box and clear any prior error message."""
        chosen = self.example.get()  # Reads selected example string from Combobox.
        self.text.delete("1.0", "end")  # Clears previous text in headline input box.
        self.text.insert("1.0", chosen)  # Inserts selected example headline.
        self.msg.set("")  # Clears any previous validation warning.

    def clear(self):
        """Reset the headline text input, prediction cards, agreement notice, and phrase breakdown table."""
        self.text.delete("1.0", "end")  # Empties input box.
        self.msg.set("")  # Clears inline status/error message.
        self.agree.configure(text="")  # Clears agreement banner.
        self.tree.delete(*self.tree.get_children())  # Removes all rows from phrase breakdown table.
        for card in (self.card_sem, self.card_rule):  # Resets both prediction cards cleanly.
            self._reset_card(card, "-" if (card is self.card_sem or self.rule is not None) else "not loaded")
        self.text.focus_set()  # Returns keyboard focus to the headline box.

    def analyze(self):
        """Validate non-empty user input and dispatch model inference on a background worker thread."""
        if self.busy or self.semantic is None:  # Ignores clicks while model is loading or already predicting.
            return  # Keeps UI responsive and thread-safe.
        headline = self.text.get("1.0", "end").strip()  # Reads user text and trims outer newlines/spaces.
        if not headline:  # Rejects empty submissions immediately before launching a thread.
            self.msg.set("Please type or select a news headline first.")  # Shows friendly inline prompt.
            return  # Stops cleanly.
        self.msg.set("")  # Clears prior validation warnings.
        self.busy = True  # Locks analyze action until worker finishes.
        self.btn.configure(state="disabled")  # Visually disables Analyze button while computing.
        self.status.set("Analyzing headline and phrase contributions...")  # Updates bottom status bar.
        threading.Thread(target=self._work, args=(headline,), daemon=True).start()  # Runs inference off main UI thread.

    def _work(self, headline: str):
        """Execute semantic prediction, phrase explanation, and optional rule-based prediction on a worker thread."""
        try:  # Captures validation errors and runtime errors cleanly to relay via self.queue.
            semantic_res = self.semantic.predict_one(headline)  # Runs full-headline Transformer + PyTorch MLP inference.
            phrase_items = self.semantic.explain_one(headline)  # Computes linguistic phrase breakdown & occlusion impacts.
            rule_res = None  # Defaults rule-based output to None if companion model is unavailable.
            if self.rule is not None:  # Scores identical headline with companion rule-based model when loaded.
                try:  # Isolates rule-based errors so semantic output is never lost.
                    rule_res = self.rule.predict_one(headline)  # Computes TF-IDF rule-based prediction.
                except Exception:  # Falls back gracefully if companion model rejects unusual input.
                    rule_res = None
            self.queue.put(("result", {"sem": semantic_res, "phrases": phrase_items, "rule": rule_res}))  # Sends bundle to UI thread.
        except ValueError as exc:  # Catches clean input-validation errors (e.g., headline > 300 chars).
            self.queue.put(("invalid", str(exc)))  # Relays user-friendly validation notice.
        except Exception as exc:  # Catches unexpected runtime errors cleanly without crashing the GUI.
            self.queue.put(("error", f"{type(exc).__name__}: {exc}"))  # Relays diagnostic message.

    def _load_models(self):
        """Load the offline semantic artifacts (and sibling Rule-Based model if found) inside a background thread."""
        try:  # Loads our trained semantic model from Semantic/semantic_artifacts/.
            from semantic_predictor import SemanticSentimentPredictor  # Imports shared inference & phrase module.
            semantic = SemanticSentimentPredictor(str(BASE / "semantic_artifacts"))  # Loads fine-tuned encoder + PyTorch MLP head.
        except Exception as exc:  # Reports clear recovery steps if artifacts or libraries are missing.
            self.queue.put((
                "fatal",
                f"Could not load the semantic model:\n{type(exc).__name__}: {exc}\n\n"
                "Make sure you have run Semantic/news-headline-semantic.ipynb first (which generates "
                "semantic_artifacts/ and semantic_predictor.py) and installed dependencies via:\n"
                "    pip install -r ../requirements.txt",
            ))
            return  # Stops loading thread on fatal semantic error.

        # Search BOTH local Semantic/ AND sibling ../Rule-Based/ for the companion rule-based pipeline (fixes original path bug!)
        rule = None  # Initializes optional companion model handle.
        rule_note = ""  # Stores optional diagnostic note if loading failed.
        candidate_rule_dirs = [BASE, BASE.parent / "Rule-Based"]  # Checks local folder and sibling Rule-Based/ folder.
        for rb_dir in candidate_rule_dirs:  # Inspects each candidate location in priority order.
            artifact = rb_dir / "news_sentiment_artifacts" / "ph_news_sentiment_pipeline.joblib"  # Expected joblib artifact path.
            if artifact.exists():  # Loads companion rule-based predictor as soon as found.
                try:  # Adds rb_dir to sys.path so news_predictor.py and news_nlp_utils.py resolve cleanly.
                    if str(rb_dir) not in sys.path:  # Avoids duplicate path entries.
                        sys.path.insert(0, str(rb_dir))  # Registers Rule-Based module path.
                    from news_predictor import NewsSentimentPredictor  # Imports companion rule-based wrapper.
                    rule = NewsSentimentPredictor(str(artifact))  # Loads TF-IDF + Logistic Regression pipeline.
                    break  # Stops searching once companion model loads cleanly.
                except Exception as exc:  # Captures companion load error cleanly without blocking semantic app.
                    rule = None  # Keeps semantic-only mode active.
                    rule_note = f" (rule-based load error: {type(exc).__name__})"  # Surfaces concise diagnostic in status bar.

        self.queue.put(("ready", (semantic, rule, rule_note)))  # Notifies Tkinter UI thread that models are ready.

    def _poll(self):
        """Drain thread-safe queue messages on the Tkinter main thread every 100 ms to update widgets safely."""
        try:  # Processes every queued message without blocking the event loop.
            while True:  # Loops until queue is empty.
                kind, payload = self.queue.get_nowait()  # Fetches next (message_type, data) pair non-blockingly.
                if kind == "ready":  # Background model loading succeeded.
                    self.semantic, self.rule, rule_note = payload  # Unpacks loaded predictors and optional warning.
                    self.btn.configure(state="normal")  # Enables Analyze button for user input.
                    ver = self.semantic.meta.get("model_version", "semantic")  # Reads saved model version tag.
                    clf_desc = self.semantic.meta.get("classifier", "Semantic Classifier")  # Reads active model description.
                    note = "" if self.rule is not None else (rule_note or " | Rule-based companion not found (semantic only)")
                    self.status.set(f"Ready — Loaded {ver}: {clf_desc}{note}")  # Displays clear loaded-model confirmation.
                    if self.rule is None:  # Marks right card clearly when companion rule-based model is absent.
                        self._reset_card(self.card_rule, "not loaded")
                elif kind == "fatal":  # Critical semantic model loading error.
                    self.status.set("Error: Semantic model could not be loaded.")  # Updates status bar cleanly.
                    messagebox.showerror("Semantic Model Unavailable", payload)  # Shows actionable error dialog.
                elif kind == "result":  # Prediction and phrase analysis finished cleanly.
                    self._show(payload)  # Populates cards, agreement indicator, phrase table, and history tab.
                    self._finish("Analysis complete.")  # Re-enables controls and updates status.
                elif kind == "invalid":  # Clean input validation error (e.g., empty or > 300 characters).
                    self.msg.set(payload)  # Shows readable inline message beneath input controls.
                    self._finish("Ready.")  # Restores ready status cleanly.
                elif kind == "error":  # Unexpected inference failure.
                    self.msg.set("Prediction error: " + payload)  # Surfaces clear diagnostic message.
                    self._finish("Ready.")  # Restores controls so user can try again.
        except queue.Empty:  # Normal exit when no more messages await processing.
            pass  # Continues event loop smoothly.
        self.after(100, self._poll)  # Reschedules queue polling 100 ms later.

    def _finish(self, status_text: str):
        """Re-enable input actions and update bottom status text after an analysis finishes."""
        self.busy = False  # Clears busy lock so user can run another headline.
        self.btn.configure(state="normal")  # Re-enables Analyze button.
        self.status.set(status_text)  # Displays updated status message.

    # ------------------------------------------------------------------ Result Rendering
    @staticmethod
    def _color(label: str) -> str:
        """Return green for Positive predictions and red for Negative predictions."""
        return POS_COLOR if label == "Positive" else NEG_COLOR  # Maps class string to semantic accent color.

    def _reset_card(self, card: dict, placeholder: str = "-"):
        """Reset a prediction card to its neutral empty state."""
        card["pred"].configure(text=placeholder, foreground=MUTED)  # Clears prominent label text.
        card["conf"].configure(text="")  # Clears confidence readout.
        card["probs"].configure(text="")  # Clears P(Pos)/P(Neg) breakdown line.
        card["meter"].set(0.0, MUTED)  # Resets confidence bar to zero width.

    def _fill_card(self, card: dict, result: dict):
        """Render predicted sentiment, confidence percentage, both class probabilities, and confidence meter on a card."""
        color = self._color(result["prediction"])  # Selects green (Positive) or red (Negative).
        p_pos = result["probabilities"].get("Positive", 0.0)  # Extracts Positive probability in [0, 1].
        p_neg = result["probabilities"].get("Negative", 0.0)  # Extracts Negative probability in [0, 1].
        card["pred"].configure(text=result["prediction"].upper(), foreground=color)  # Displays bold uppercase verdict.
        card["conf"].configure(text=f"{result['confidence']:.1%} confident", foreground=color)  # Displays exact confidence %.
        card["probs"].configure(text=f"P(Positive): {p_pos:.1%}   |   P(Negative): {p_neg:.1%}", foreground=MUTED)  # Displays both probs.
        card["meter"].set(result["confidence"], color)  # Updates visual confidence meter bar.

    def _show(self, data: dict):
        """Populate both prediction cards, agreement indicator, phrase-by-phrase breakdown table, and History log."""
        sem, phrases, rule = data["sem"], data["phrases"], data["rule"]  # Unpacks worker payload cleanly.
        self._fill_card(self.card_sem, sem)  # Renders primary semantic deep learning prediction.

        if rule is not None:  # Renders companion rule-based prediction and agreement status when available.
            self._fill_card(self.card_rule, rule)  # Fills right-hand comparison card.
            if rule["prediction"] == sem["prediction"]:  # Both models reached the same verdict.
                self.agree.configure(text="Both models agree on overall sentiment.", foreground=MUTED)
            else:  # Highlights model disagreement—common on negation and contrast headlines!
                self.agree.configure(
                    text="Models disagree — inspect the phrase breakdown below to see how context affects sentiment.",
                    foreground=WARN_COLOR,
                )
        else:  # Keeps right card cleanly marked as unloaded when Rule-Based artifacts are absent.
            self._reset_card(self.card_rule, "not loaded")
            self.agree.configure(text="")

        # Populate upgraded 6-column Phrase-by-Phrase Semantic Breakdown table in reading order
        self.tree.delete(*self.tree.get_children())  # Clears previous headline's phrases cleanly.
        for item in sorted(phrases, key=lambda d: d["order"]):  # Inserts phrases left-to-right in sentence order.
            tag = "pos" if item["pushes_toward"] == "Positive" else "neg"  # Chooses green/red row highlight by push direction.
            alone_lbl = item.get("alone_label", "Positive" if item["alone_pos"] >= 0.5 else "Negative")  # Standalone class.
            alone_neg = item.get("alone_neg", 1.0 - item["alone_pos"])  # Standalone Negative probability.
            self.tree.insert(  # Adds complete Task 11C row to the Treeview table.
                "",
                "end",
                tags=(tag,),
                values=(
                    item["phrase"],  # Extracted linguistic phrase.
                    alone_lbl,  # Individual phrase sentiment.
                    f"{item['alone_pos']:.1%}",  # Individual Positive probability.
                    f"{alone_neg:.1%}",  # Individual Negative probability.
                    f"{item['impact']:+.3f}",  # Contextual occlusion impact on full headline.
                    item["pushes_toward"],  # Overall direction pushed ('Positive' or 'Negative').
                ),
            )

        # Append newest prediction to top of the History tab
        sem_ppos = sem["probabilities"].get("Positive", 0.0)  # Extracts Positive probability for session history log.
        self.history_tree.insert(  # Inserts newest record at row index 0 (most recent first).
            "",
            0,
            values=(
                dt.datetime.now().strftime("%H:%M:%S"),  # Local analysis timestamp.
                sem["headline"],  # Normalized headline text.
                sem["prediction"],  # Semantic prediction label.
                f"{sem['confidence']:.1%}",  # Semantic confidence %.
                f"{sem_ppos:.1%}",  # Semantic P(Positive) %.
                rule["prediction"] if rule else "-",  # Companion rule-based label if loaded.
                f"{rule['confidence']:.1%}" if rule else "-",  # Companion rule-based confidence if loaded.
            ),
        )

    # ------------------------------------------------------------------ History Export & Cleanup
    def export_history(self):
        """Export all predictions recorded in the current session to a UTF-8 CSV spreadsheet."""
        rows = [self.history_tree.item(item_id, "values") for item_id in self.history_tree.get_children()]  # Reads all table rows.
        if not rows:  # Alerts user politely if no headlines have been analyzed yet.
            messagebox.showinfo("Nothing to Export", "Analyze at least one headline before exporting history.")
            return  # Exits export handler cleanly.
        path = filedialog.asksaveasfilename(  # Opens native Save As dialog defaulting to CSV format.
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="sentiment_history.csv",
        )
        if not path:  # Cancels cleanly if user closes the file dialog without selecting a path.
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:  # Writes Excel-compatible UTF-8 CSV with BOM.
            writer = csv.writer(handle)  # Instantiates standard CSV writer.
            writer.writerow(["time", "headline", "semantic", "semantic_confidence", "semantic_p_pos", "rule_based", "rule_confidence"])
            writer.writerows(rows)  # Writes every logged prediction row.
        self.status.set(f"Exported {len(rows)} prediction(s) to {path}")  # Confirms export path in status bar.

    def clear_history(self):
        """Remove all rows from the session prediction history table."""
        self.history_tree.delete(*self.history_tree.get_children())  # Deletes every item currently in the History Treeview.
        self.status.set("Prediction history cleared.")  # Confirms cleanup in status bar.


if __name__ == "__main__":  # Runs the desktop GUI only when app_tk.py is executed directly (safe for module import tests!).
    try:  # Enables crisp font rendering on high-DPI Windows displays when supported.
        import ctypes  # Imports Windows DLL binding module.
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Requests system-level DPI scaling on Windows.
    except Exception:  # Safely ignores DPI call on non-Windows platforms or older OS builds.
        pass
    App().mainloop()  # Instantiates the Tkinter window and starts the desktop GUI event loop.
