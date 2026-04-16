# drift-lens Launch Posts

Post in this order, **24 hours apart**.

---

## 1. r/MachineLearning (Day 1)

**Title:** Show r/ML: I built a tool that detects embedding drift before your accuracy drops

**Flair:** [Project]

**Body:**

Hey r/ML,

I kept running into the same problem: model accuracy looks fine for days, then suddenly craters. By the time precision drops, the embedding space has already shifted. Accuracy is a lagging indicator — embeddings are the leading one.

So I built **drift-lens** — an open-source Python tool that detects embedding space drift *before* your metrics degrade.

**What makes it different from Evidently/Arize:**
- **Topological drift detection** via persistent homology (Ripser). This catches cluster merges, splits, and structural collapse that statistical tests completely miss.
- **Three detection methods**: Fréchet Embedding Distance (like FID but for arbitrary embeddings), kernel MMD with permutation p-values, and the topology detector.
- **Zero infrastructure**: `pip install drift-lens` → 5 lines of code → done. No cloud, no API keys, no YAML. Snapshots are flat parquet files.

**The demo that sold me on building this:**
The dashboard shows a synthetic 14-day scenario where drift-lens fires an alert on Day 8. Accuracy doesn't drop until Day 12. That's a **4-day early warning** from 5 lines of code.

[Demo GIF]

**Quick start:**
```python
from drift_lens import EmbeddingLogger, DriftDetector

logger = EmbeddingLogger(path="./embeddings", window="1d")
logger.log(embeddings)

detector = DriftDetector(method="frechet")
result = detector.compare(baseline, current)
print(f"Drift: {result.drift_score:.3f} | Alert: {result.is_drift}")
```

GitHub: https://github.com/PRAFULREDDYM/drift-lens
PyPI: `pip install drift-lens`

Happy to answer questions about the math, the topology approach, or why I think embedding monitoring is criminally underserved.

---

## 2. Hacker News (Day 2)

**Title:** Show HN: drift-lens – detect embedding space drift before your metrics do

**URL:** https://github.com/PRAFULREDDYM/drift-lens

**Comment (post immediately after submission):**

Hi HN,

drift-lens is an open-source Python tool for detecting drift in embedding spaces. The core idea: if you monitor your model's embeddings directly, you can catch distribution shifts days before downstream accuracy degrades.

Three detection methods:
1. Fréchet Embedding Distance — same math as FID, applied to arbitrary embeddings
2. Maximum Mean Discrepancy — kernel-based, non-parametric, with permutation p-values
3. Persistent homology — computes topological features (connected components, loops) via Vietoris-Rips filtration and compares them with Wasserstein distance. This is the unusual one — it detects structural changes like cluster merges that FED/MMD miss.

Install: `pip install drift-lens`

No cloud, no API keys, no infrastructure. Snapshots are just parquet files. Ships with a Streamlit dashboard for visual exploration.

The synthetic demo shows it catching drift 4 days before accuracy drops.

---

## 3. r/LocalLLaMA (Day 3)

**Title:** drift-lens — open-source tool to monitor your local embedding model in production

**Body:**

If you're running a local embedding model (sentence-transformers, Nomic, etc.) in production, how do you know when the distribution of your embeddings has shifted?

I built **drift-lens** to solve this. It monitors the embedding space itself — not accuracy, not loss — and fires an alert when the statistical or topological structure changes.

**Why this matters for local models:**
- You swap from `all-MiniLM-L6-v2` to `nomic-embed-text` → drift-lens quantifies exactly how different the spaces are
- Your input data distribution shifts over time → drift-lens catches it before your RAG pipeline starts returning garbage
- You fine-tune your model → drift-lens tells you if the new embeddings have collapsed or diverged

**Usage:**
```python
from drift_lens import EmbeddingLogger, DriftDetector

logger = EmbeddingLogger(path="./embeddings", window="1d")
logger.log(embeddings)  # works with numpy, torch, lists

detector = DriftDetector(method="topology")  # catches structural changes
result = detector.compare(baseline, current)
```

Runs entirely local. No API keys, no cloud. Just `pip install drift-lens`.

GitHub: https://github.com/PRAFULREDDYM/drift-lens

---

## 4. Twitter/X Thread (Day 4)

**Thread (5 tweets):**

---

**Tweet 1 (lead with GIF):**

Your model's accuracy looks fine.

But the embedding space has been drifting for 4 days.

I built drift-lens: open-source embedding drift detection that fires BEFORE your metrics drop.

[ATTACH DEMO GIF]

🧵👇

---

**Tweet 2:**

The problem: accuracy, precision, recall — they're all LAGGING indicators.

By the time they drop, your embeddings shifted days ago.

drift-lens watches the leading indicator: the embedding space itself.

3 detection methods. 5 lines of code. Zero API keys.

---

**Tweet 3:**

What makes this different:

1️⃣ Fréchet Embedding Distance (the "FID for embeddings")
2️⃣ Kernel MMD with permutation p-values
3️⃣ Topological drift detection via persistent homology

That third one? No other production tool does it. It catches cluster merges, splits, and structural collapse.

---

**Tweet 4:**

```python
from drift_lens import DriftDetector

detector = DriftDetector(method="topology")
result = detector.compare(baseline, current)
# result.drift_score → 0.0 to 1.0
```

Install: pip install drift-lens
No cloud. No YAML. Just parquet files.

---

**Tweet 5:**

If you're running embeddings in production — RAG, search, recommendations, classification — you should be monitoring the embedding space directly.

⭐ GitHub: https://github.com/PRAFULREDDYM/drift-lens

Built as an open-source tool for the MLOps community.

Tag someone who's been bitten by silent drift 👇

@chiphuyen @_philschmid @Nils_Reimers @hamaborsa @sh_reya

---

## Posting Checklist

- [ ] Ensure demo GIF is pushed to repo (`assets/demo.gif`)
- [ ] Verify `pip install drift-lens` works from PyPI
- [ ] Day 1: Post to r/MachineLearning
- [ ] Day 2: Post to Hacker News
- [ ] Day 3: Post to r/LocalLLaMA
- [ ] Day 4: Post Twitter/X thread
- [ ] Respond to every comment in the first 2 hours (critical for algorithm boost)
