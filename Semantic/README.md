# Semantic Sentiment Analysis (`Semantic/`)

Context-aware Deep Learning sentiment analysis pipeline and Tkinter desktop application for **Philippines News Headlines (2014–2023)**.

---

## Notebook Overview (`news-headline-semantic.ipynb` — Exactly 6 Markdown Sections)

All theory (Weeks 04–07 lectures), Exploratory Data Analysis, model comparisons, semantic stress tests, phrase-by-phrase analysis, and setup instructions are organized directly inside **6 clean Markdown sections** (strictly 6 Outline headings; 24 total cells: 6 Markdown + 18 concise Code cells powered by `semantic_utils.py`, with a **Graph Interpretation** rendered directly beneath every graph):

1. **1. Feedforward Neural Networks and Perceptrons** (Week 04: setup, data/embedding loader, activations, BCE vs. MSE loss, Perceptron `AND`/`OR`/`XOR`, Forward Pass, `NumpyMLP` Backpropagation & Gradient Check).
2. **2. Optimization and Training Dynamics in Deep Learning** (Week 05: from-scratch `SGD`, `SGD + Momentum`, `Adam`, Learning-Rate sweeps, Overfitting, Early Stopping, Dropout & scikit-learn optimizer curves).
3. **3. Convolutional Neural Networks (CNNs)** (Week 06: convolution, padding/stride, pooling, receptive fields, CNNs vs. Transformers for news text, and a runnable 1D-CNN vs. Transformer comparison).
4. **4. Transfer Learning and Pretrained Models** (Week 07: Feature Extraction vs. Partial Fine-Tuning, pretrained vision vs. NLP backbones, cached embeddings, PCA projection & Negation Geometry Probe).
5. **5. Project Methodology, Comprehensive EDA, Model Comparison & Model Evaluation** (full 64k EDA & 3-class RoBERTa bias audit, 7-model comparison, Week 07 Partial Fine-Tuning of `distilbert-base-uncased` Top 2/6 layers + CLS/Mean PyTorch BN/Dropout MLP + Soft-Target Distillation, and untouched Before-vs-After test evaluation on `11,545` headlines).
6. **6. Project Setup, Implementation, Semantic Testing, Phrase Analysis & User Guide** (offline packaging, 24-sentence 4-way & 34-sentence 10-category stress tests, linguistic phrase breakdown, `human_label_sample.csv`, robustness, Responsible AI & `app_tk.py` setup/guide).

---

## Verified Highlights (`semantic-3.0`)

- **Untouched Test Set (`11,545` headlines from 100% of all `64,341` unique clean headlines, SST-2 teacher agreement):** Accuracy **`85.60% -> 94.37%` (+8.77%)**, Macro-F1 **`85.36% -> 94.28%` (+8.92%)**, ROC-AUC **`0.9336 -> 0.9863` (+0.0527)** at **`9.8 ms/headline`** CPU latency (disk reload parity **`9.31e-10`**).
- **Independent Semantic Stress Tests:** 24-sentence stress test **`50.0% -> 70.8%` (`17/24`)**; 34-sentence 10-category stress test **`97.1%` (`33/34`)** vs. companion Rule-Based **`52.9%` (`18/34`)**.
- **Simple Week-5-Style Visuals & Interpretations:** Clean white canvas, hidden top/right spines, light dashed grid, Blue/Orange/Green/Red palette, exact `Underfitting | Right Fit (Early Stopping) | Overfitting` curves, and a bold **Graph Interpretation** directly beneath all `21/21` charts.
- **Upgraded Tkinter App (`app_tk.py`):** Auto-loads `semantic-3.0` + companion `../Rule-Based/news_sentiment_artifacts/`, shows explicit `P(Positive)` & `P(Negative)`, 6-column linguistic phrase breakdown, scrollable History with CSV export, and scrollable About tab.

---

## Running Locally

```powershell
# From repository root (Electives/)
conda activate Luiz
python -m pip install -r requirements.txt

# Launch the offline Tkinter desktop GUI
cd Semantic
python app_tk.py
```
