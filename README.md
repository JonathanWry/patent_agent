# Patent Prior-Art Agent

Patent search and prior-art analysis built on PANORAMA.

This repo has two purposes:

1. a **product-style patent agent** for free-text patent search and QA
2. a **benchmark pipeline** for evaluating retrieval methods on PANORAMA `PAR4PC`

---

## 1. What the system does

### Product mode: `Our Patent Agent`

Input:
- patent claim text
- invention description
- vague patent search query
- follow-up question over previous results

Output:
- related patents
- optional evidence snippets
- grounded answer
- multi-turn follow-up using previous retrieved results

### Benchmark mode: `Benchmark`

Input:
- one PANORAMA `PAR4PC` case

Output:
- ranked A-H prior-art candidates
- claim decomposition
- evidence chart
- verification
- metrics against gold answers

---

## 2. Main methods

This project exposes **two product-level methods** in the UI.

### A. Normal RAG baseline

This is the simple comparison path.

Pipeline:

```text
persistent local index
-> retrieve top patents
-> optional evidence snippets
```

Current implementation details:
- patent source: `Persistent local index`
- retrieval: `local-embedding` / `PatentSBERTa`
- no planner
- no second-stage reranker
- no grounded answer synthesis
- no answer verification

Purpose:
- serve as the plain patent-RAG baseline
- show what a simpler retrieval-only system returns

### B. Our optimized patent agent

This is the main product method.

Pipeline:

```text
persistent local index coarse recall
-> linear-patent-reranker second-stage rerank
-> conversational planner / working-set reuse
-> evidence extraction
-> grounded answer
-> answer verification
```

What it adds on top of the baseline:

1. **second-stage learned reranking**
   - uses `linear-patent-reranker`
   - trained from PANORAMA benchmark data

2. **conversational planner**
   - decides whether to retrieve again or reuse the current working set

3. **working-set reuse**
   - follow-up questions operate over previous top results when appropriate

4. **evidence extraction**
   - selects supporting title / abstract / claim snippets

5. **grounded answer generation**
   - answers from retrieved evidence instead of only listing patents

6. **answer verification**
   - checks whether the answer is supported by retrieved evidence

This is the main method to demo.

### Why this split exists

We keep the product comparison intentionally simple:

- `Normal RAG baseline`
  - answers the question: "what does a simpler patent retrieval system return?"
- `Our optimized patent agent`
  - answers the question: "what changes when we add patent-specific reranking, context reuse, evidence handling, and verification?"

This keeps the UI comparison coherent. The baseline is intentionally narrow. The optimized path is the full system.

### What each optimized step is for

#### Persistent local index coarse recall

Purpose:
- retrieve candidate patents quickly from a larger local pool
- avoid rebuilding embeddings or indices every run

#### Linear patent reranker

Purpose:
- rerank the recalled candidate patents using benchmark-validated patent features
- move beyond pure semantic similarity

Current feature set:
- `dense_score`
- `bm25_score`
- `field_lexical_score`

#### Conversational planner / working-set reuse

Purpose:
- support follow-up patent questions
- reuse current results when the user is filtering or comparing prior results

#### Evidence extraction

Purpose:
- identify the most relevant title / abstract / claim snippets
- show why a patent was retrieved

#### Grounded answer

Purpose:
- answer from retrieved evidence rather than only listing patents

#### Answer verification

Purpose:
- indicate whether the answer appears supported by the retrieved evidence

---

## 3. What each component means

### `Persistent local index`

This is **not** a model.

It is a prebuilt local patent search backend containing:
- `index.faiss`
- `metadata.parquet`
- `manifest.json`

It is used for fast coarse recall in product mode.

### `local-embedding`

This is a retrieval method.

Current implementation:
- model: `AI-Growth-Lab/PatentSBERTa`

Used as:
- the product baseline retriever
- the stable benchmark baseline

### `linear-patent-reranker`

This is a learned reranking method.

It is trained on benchmark-derived patent features and currently uses:
- `dense_score`
- `bm25_score`
- `field_lexical_score`

Used as:
- the main learned benchmark method
- the second-stage reranker inside the optimized product pipeline

---

## 4. Why there is a benchmark mode

`Benchmark` exists to answer a different question from the product UI.

### Product mode asks:

> Can a user type a patent idea or claim and get useful related patents plus grounded analysis?

### Benchmark mode asks:

> Do our retrieval methods actually improve prior-art ranking on labeled PANORAMA cases?

So benchmark mode is used for:
- method comparison
- retrieval metrics
- ablations
- validating rerankers before moving them into the product pipeline

Benchmark mode uses PANORAMA `PAR4PC`, where each case has:
- one target claim
- candidate prior-art patents `A-H`
- gold answers

In short:

- product mode is for user-facing patent search and QA
- benchmark mode is for validating retrieval improvements before or while moving them into the product path

---

## 5. Current benchmark setup

### Benchmark baseline

`PatentSBERTa baseline`

Pipeline:

```text
target claim
-> local-embedding over A-H candidates
-> ranked candidate letters
```

### Benchmark improved method

`Our learned reranker`

Pipeline:

```text
target claim
-> patent features for A-H candidates
-> linear-patent-reranker
-> ranked candidate letters
```

Current best learned reranker setup:
- train split: HF `train`
- train rows: `200`
- features:
  - `dense_score`
  - `bm25_score`
  - `field_lexical_score`

Representative `validation-100` result:

| Method | hit@1 | hit@3 | recall@3 | exact@|gold| |
|---|---:|---:|---:|---:|
| `local-embedding` | 0.590 | 0.860 | 0.802 | 0.470 |
| `linear-patent-reranker` | 0.600 | 0.910 | 0.850 | 0.510 |

Interpretation:
- `local-embedding` is the stable baseline
- `linear-patent-reranker` is the current learned improvement

---

## 6. Setup from scratch

### Requirements

Install first:
- `git`
- `conda`
- Python `3.10`

Optional:
- OpenAI API key for LLM-backed answer generation / decomposition / verification

### Clone repos

Clone this repo:

```bash
git clone <your-patent-agent-repo-url>
cd patent-agent
```

Clone PANORAMA as a sibling directory:

```bash
cd ..
git clone https://github.com/LGAI-Research/PANORAMA.git
cd patent-agent
```

Expected layout:

```text
workspace/
  PANORAMA/
  patent-agent/
```

### Create environment

```bash
conda env create -f environment.yml
conda activate patent-agent
```

If needed:

```bash
pip install -r requirements.txt
```

### Configure secrets

```bash
cp .env.example .env
```

Then edit `.env` if you want OpenAI-backed features:

```text
OPENAI_API_KEY=your_key_here
PATENT_AGENT_MODEL=gpt-4o-mini
```

---

## 7. Build the persistent local index

The product UI expects a local patent index.

Recommended demo index:

```bash
python -m src.build_patent_index \
  --pool-source combined \
  --hub-rows-per-split 50 \
  --index-dir data/indexes/par4pc_patentsberta_demo
```

This creates:

```text
data/indexes/par4pc_patentsberta_demo/
  index.faiss
  metadata.parquet
  manifest.json
```

What this means:
- `combined` = local PANORAMA sample patents + a small Hub slice
- good enough for demo and UI testing

Larger index example:

```bash
python -m src.build_patent_index \
  --pool-source hub \
  --hub-rows-per-split 2000 \
  --index-dir data/indexes/par4pc_patentsberta_large
```

---

## 8. Prebuild the learned reranker

If you want the improved method to be ready before the UI starts:

```bash
python -m src.train_linear_patent_reranker \
  --mode train-default-model \
  --splits train \
  --max-rows-per-split 200
```

This saves:
- `data/models/linear_patent_reranker_patentsberta_train200_3feat.joblib`
- `data/models/linear_patent_reranker_patentsberta_train200_3feat.json`

After that, the reranker loads from disk instead of retraining.

Optional feature cache prebuild:

```bash
python -m src.feature_cache \
  --source hf \
  --splits train \
  --max-rows-per-split 200 \
  --namespace linear_train_200cases

python -m src.feature_cache \
  --source hf \
  --splits validation \
  --max-rows-per-split 100 \
  --namespace scan_eval_100cases
```

---

## 9. How to run the system

Always activate the environment first:

```bash
cd "/Users/jonathanwang/Desktop/Emory/Year_4_Sem_2/CS 329/patent-agent"
conda activate patent-agent
```

### Launch the UI

```bash
./scripts/run_app.sh
```

### Product UI usage

Mode:
- `Our Patent Agent`

Then choose one of:
- `Normal RAG baseline`
- `Our optimized patent agent`
- `Side-by-side comparison`

Recommended for demo:
- `Side-by-side comparison`

What to expect:
- left / baseline: top patents only, optional snippets
- right / optimized: reranked results + grounded answer + verification

### Benchmark UI usage

Mode:
- `Benchmark`

Then choose:
- `PatentSBERTa baseline`
- or `Our learned reranker`

Recommended for demo:
1. show `PatentSBERTa baseline`
2. switch to `Our learned reranker`
3. compare outputs on the same case

---

## 10. Main experiment commands

### Local benchmark evaluation

```bash
python -m src.evaluate_par4pc --retrieval-method local-embedding
python -m src.evaluate_par4pc --retrieval-method linear-patent-reranker
```

### HF validation evaluation

```bash
python -m src.evaluate_par4pc_hf \
  --splits validation \
  --max-rows-per-split 100 \
  --methods local-embedding linear-patent-reranker
```

### Retrieval comparison table

```bash
python -m src.compare_retrieval --output outputs/retrieval_comparison.csv
```

### Learned reranker scan

```bash
python -m src.scan_linear_reranker_configs \
  --train-rows 50 100 200 \
  --eval-rows 100 \
  --output outputs/linear_reranker_scan.csv
```

### Free-text single-turn demo

```bash
python -m src.run_free_text_demo
```

### Free-text conversation demo

```bash
python -m src.run_conversation_demo
```

---

## 11. What works without OpenAI

Without `OPENAI_API_KEY`, these still work:
- product baseline retrieval
- optimized product retrieval
- benchmark evaluation
- persistent index build
- evidence extraction
- heuristic grounded answer
- heuristic answer verification
- conversational planner and working-set reuse

OpenAI is only needed for optional LLM-backed paths such as:
- LLM grounded answer
- LLM claim decomposition
- LLM answer verification
- `openai-embedding`
- `llm-rerank`

---

## 12. Common problems

### Problem: `ModuleNotFoundError: No module named 'torchvision'`

This usually means the conda environment is not the one expected by the repo.

Fix:

```bash
conda activate patent-agent
pip install -r requirements.txt
```

Then launch with:

```bash
./scripts/run_app.sh
```

Do not start Streamlit from a random Python environment.

### Problem: `import app` fails because `streamlit` is missing

Same cause: wrong environment.

Fix:

```bash
conda activate patent-agent
```

### Problem: the UI says the persistent index is missing

Build it first:

```bash
python -m src.build_patent_index \
  --pool-source combined \
  --hub-rows-per-split 50 \
  --index-dir data/indexes/par4pc_patentsberta_demo
```

### Problem: learned reranker is slow the first time

Prebuild it:

```bash
python -m src.train_linear_patent_reranker \
  --mode train-default-model \
  --splits train \
  --max-rows-per-split 200
```

---

## 13. What the project still does not claim

- This is not legal advice.
- The product path is a technical prior-art exploration tool.
- The benchmark is strongest on PANORAMA `PAR4PC`.
- Product QA quality still depends on the coverage of the local index.

---

## 14. Git / repo notes

Ignored by git:
- `.env`
- `outputs/`
- `data/indexes/`
- `data/models/`
- `data/cache/`
- local Streamlit secrets

To push:

```bash
git remote add origin <your-github-repo-url>
git push -u origin main
```

---

## 15. Extra teammate docs

For teammates who need the system explanation rather than the run commands:

- method and component overview:
  - [docs/METHODS_OVERVIEW.md](/Users/jonathanwang/Desktop/Emory/Year_4_Sem_2/CS%20329/patent-agent/docs/METHODS_OVERVIEW.md)
- handoff / UI / evaluation notes:
  - [docs/TEAM_HANDOFF.md](/Users/jonathanwang/Desktop/Emory/Year_4_Sem_2/CS%20329/patent-agent/docs/TEAM_HANDOFF.md)
