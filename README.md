# Patent Prior-Art Agent

Conversational patent search and prior-art analysis built on PANORAMA.

This repository supports:

- PANORAMA `PAR4PC` benchmark evaluation
- persistent local patent indexing with FAISS
- free-text patent search
- grounded QA over retrieved patent evidence
- multi-turn follow-up handling with working-set reuse
- benchmark claim decomposition, evidence extraction, and verification

This README includes:

1. full setup from scratch
2. data layout
3. index building
4. all main experiment commands
5. demo / UI steps

## 1. Requirements

Install these first:

- `git`
- `conda`
- Python `3.10` compatible environment

Optional:

- OpenAI API key for LLM-grounded answers and LLM-backed benchmark steps

## 2. Clone Repositories

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

Recommended directory layout:

```text
workspace/
  PANORAMA/
  patent-agent/
```

The code assumes local PANORAMA sample benchmark files live at:

```text
../PANORAMA/data/benchmark/par4pc
```

## 3. Create Environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate patent-agent
```

If you prefer manual install:

```bash
pip install -r requirements.txt
```

## 4. Configure Secrets

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
OPENAI_API_KEY=your_key_here
PATENT_AGENT_MODEL=gpt-4o-mini
```

Notes:

- `.env` is gitignored
- without `OPENAI_API_KEY`, the system still works using heuristic grounded answers
- LLM-grounded answer, LLM claim decomposition, and LLM verification require the key

## 5. Verify Basic Setup

Run syntax and import checks:

```bash
python -m compileall app.py src
python -c "import app; print('app import ok')"
```

Expected:

- no syntax errors
- prints `app import ok`

## 6. Build a Persistent Local Index

This project supports free-text search over a persistent FAISS index.

### Recommended demo index

Build the combined demo index:

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

### Larger local index

If you want a larger index:

```bash
python -m src.build_patent_index \
  --pool-source hub \
  --hub-rows-per-split 2000 \
  --index-dir data/indexes/par4pc_patentsberta_large
```

Full split:

```bash
python -m src.build_patent_index \
  --pool-source hub \
  --hub-rows-per-split 0 \
  --index-dir data/indexes/par4pc_patentsberta_full
```

Notes:

- `combined` = local PANORAMA sample patents + Hub slice
- `hub-rows-per-split 0` means full train/validation/test
- first build can take time because the embedding model is loaded and the patent texts are encoded

## 7. Main Experiment Commands

This section lists all major experiments and evaluation commands.

### 7.1 Benchmark retrieval evaluation

Default:

```bash
python -m src.evaluate_par4pc
```

PatentSBERTa local embedding:

```bash
python -m src.evaluate_par4pc --retrieval-method local-embedding
```

Patent-specific hybrid reranking:

```bash
python -m src.evaluate_par4pc --retrieval-method hybrid-coverage
```

Learned linear patent reranker:

```bash
python -m src.evaluate_par4pc --retrieval-method linear-patent-reranker
```

Prebuild the cached linear reranker model:

```bash
python -m src.train_linear_patent_reranker \
  --mode train-default-model \
  --splits train \
  --max-rows-per-split 200
```

Prebuild feature caches used by the learned reranker experiments:

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

BM25:

```bash
python -m src.evaluate_par4pc --retrieval-method bm25
```

Experimental methods:

```bash
python -m src.evaluate_par4pc --retrieval-method local-cross-encoder
python -m src.evaluate_par4pc --retrieval-method openai-embedding
python -m src.evaluate_par4pc --retrieval-method llm-rerank
```

Current pilot result on bundled local `PAR4PC` samples:

```text
local-embedding retrieval
hit@1: 0.800
hit@3: 1.000
recall@3: 1.000
exact@|gold|: 0.700
```

Current interpretation:

- `local-embedding` with `AI-Growth-Lab/PatentSBERTa` is still the strongest benchmark default
- `hybrid-coverage` is a patent-specific ablation that combines dense retrieval, BM25, and limitation coverage, but it does not yet beat PatentSBERTa on the bundled local sample set
- `linear-patent-reranker` is the current learned benchmark experiment; it is trained on a cached HF train slice and is not the default demo path
- the cached model is stored under `data/models/` so later processes can load it without retraining

### 7.2 Retrieval comparison table

```bash
python -m src.compare_retrieval --output outputs/retrieval_comparison.csv
```

This compares:

- `bm25`
- `hybrid-coverage`
- `local-embedding` with patent-domain and general-domain models
- `local-cross-encoder` rerankers

### 7.3 Larger-split HF evaluation

Evaluate on a larger Hugging Face `PAR4PC` validation slice:

```bash
python -m src.evaluate_par4pc_hf \
  --splits validation \
  --max-rows-per-split 100 \
  --methods local-embedding linear-patent-reranker
```

This is the main way to check whether a retrieval method still holds up beyond the bundled local examples.

### 7.4 Patent-specialized ablation

Run a quick component ablation over the patent-specialized reranker:

```bash
python -m src.ablate_patent_specialized \
  --splits validation \
  --max-rows-per-split 30 \
  --output outputs/patent_specialized_ablation.csv
```

### 7.5 Learned linear reranker experiments

Forward-selection table over patent-specialized features:

```bash
python -m src.train_linear_patent_reranker \
  --max-rows-per-split 100 \
  --output outputs/linear_reranker_forward_selection_100.csv
```

Targeted single-config evaluation for the current best linear subset:

```bash
python -m src.train_linear_patent_reranker \
  --mode single \
  --max-rows-per-split 100 \
  --feature-names dense_score bm25_score field_lexical_score \
  --class-weight none \
  --c-value 4.0
```

Train-size / feature-set scan:

```bash
python -m src.scan_linear_reranker_configs \
  --train-rows 50 100 200 \
  --eval-rows 100 \
  --output outputs/linear_reranker_scan.csv
```

If you prebuilt the matching feature caches first, repeated scans reuse the parquet caches under `data/cache/features/` instead of recomputing all patent-specialized feature vectors.

When trained on a separate HF train slice and evaluated on `validation-100`, the learned linear reranker currently:

- best current config is `train_rows=200` with `dense_score + bm25_score + field_lexical_score`
- this reaches `hit@1=0.600`, `hit@3=0.910`, `recall@3=0.850`, `exact@|gold|=0.510` on `validation-100`
- compared with pure `local-embedding` at `hit@1=0.590`, `hit@3=0.860`, `recall@3=0.802`, `exact@|gold|=0.470`
- it remains the preferred experimental benchmark path over the full hand-tuned `patent-specialized` score

### 7.6 Benchmark report generation

Heuristic / no-LLM:

```bash
python -m src.run_demo --retrieval-method local-embedding --output outputs/demo_report_verified.md
```

With LLM-backed benchmark steps:

```bash
python -m src.run_demo \
  --retrieval-method llm-rerank \
  --llm-decompose \
  --llm-verify \
  --output outputs/demo_report_llm.md
```

### 7.7 Single-turn free-text QA

Heuristic grounded answer:

```bash
python -m src.run_free_text_demo
```

With OpenAI grounded answer:

```bash
python -m src.run_free_text_demo --llm-answer
```

### 7.5 Multi-turn conversation demo

```bash
python -m src.run_conversation_demo
```

This tests:

1. new search
2. aspect filter on current results
3. compare previous results
4. combination exploration with context-enriched retrieval

## 8. What Each Mode Is For

### Benchmark Analysis

This is for labeled PANORAMA evaluation.

It uses a known `PAR4PC` case and does:

```text
target claim
-> rank A-H candidate patents
-> decompose claim
-> extract evidence
-> verify evidence
-> render report
```

Use this mode for:

- project evaluation
- retrieval metrics
- structured evidence demo

### Free-text Search

This is the user-facing patent search / QA mode.

It does:

```text
user query or claim
-> decide whether to retrieve new or reuse context
-> retrieve or rerank patents
-> gather evidence snippets
-> answer from evidence
```

Use this mode for:

- vague patent search
- similar patent search
- aspect filtering
- follow-up questions
- combination exploration

## 9. Run the Streamlit UI

Launch:

```bash
./scripts/run_app.sh
```

### Recommended demo configuration

Set:

- `Mode = Free-text Search`
- `Retrieval method = local-embedding`
- `Free-text patent pool = Persistent local index`
- `Persistent index directory = data/indexes/par4pc_patentsberta_demo`

Then click:

- `Use example query`

After that, try follow-up questions such as:

```text
Which of those also includes access control for the requested information?
```

```text
Compare the top two patents for participant context and profile handling.
```

```text
If I combine that with smart invitations, what related patents should I inspect next?
```

### Recommended benchmark configuration

Set:

- `Mode = Benchmark Analysis`
- `Retrieval method = local-embedding`

Then:

- select a `PAR4PC` case
- click `Analyze Benchmark Case`

## 10. Manual Test Procedure

If you want a minimal end-to-end test on a fresh machine, run these in order:

```bash
python -m compileall app.py src
python -m src.build_patent_index --pool-source combined --hub-rows-per-split 50 --index-dir data/indexes/par4pc_patentsberta_demo
python -m src.evaluate_par4pc --retrieval-method local-embedding
python -m src.run_free_text_demo
python -m src.run_conversation_demo
./scripts/run_app.sh
```

This covers:

- code integrity
- index build
- benchmark retrieval
- single-turn QA
- multi-turn QA
- UI demo

## 11. Input Guidance

Best input:

- full patent claim text, e.g. `1. A method comprising: ...`

Also supported:

- invention description
- vague technical query
- question over current retrieved results

Examples:

```text
patents about event participants getting personalized information during a live event
```

```text
find patents similar to systems and methods for presenting information extracted from one or more data sources to event participants
```

```text
Which of those also includes access control for the requested information?
```

## 12. What Works Without OpenAI

Without `OPENAI_API_KEY`, these still work:

- benchmark retrieval
- benchmark report generation with heuristic decomposition / verification
- persistent index build
- free-text retrieval
- heuristic grounded QA
- multi-turn planner and working-set reuse

These require OpenAI:

- `Use LLM grounded answer`
- `Use LLM claim decomposition`
- `Use LLM evidence verification`
- `openai-embedding`
- `llm-rerank`

Patent-specific retrieval experiments available without OpenAI:

- `hybrid-coverage`
- `local-embedding`
- `bm25`

## 13. Notes and Limitations

- This is a technical prior-art exploration tool, not legal advice.
- The planner for conversational follow-up is heuristic, not fully autonomous.
- The benchmark evaluation is strongest on `PAR4PC`; free-text QA is a demo-oriented extension.
- The quality of free-text search depends on the persistent index coverage.
- The current patent-specific `hybrid-coverage` reranker addresses real patent-RAG pain points, but it still needs tuning before it can replace PatentSBERTa as the benchmark default.

## 14. Preparing for GitHub

Ignored by git:

- `.env`
- `.venv/`
- `outputs/`
- `data/indexes/`
- caches
- local Streamlit secrets

To push this repo:

```bash
git remote add origin <your-github-repo-url>
git push -u origin main
```
