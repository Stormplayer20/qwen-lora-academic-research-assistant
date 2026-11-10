# Lightweight Academic Research Assistant with LoRA-Tuned Qwen

**An undergraduate AI research project developed during Rider University's 2026 MacMillan Summer Fellowship.**

This repository contains the Python application, 500-example query-rewriting dataset, dataset generator, and the **original Google Colab LoRA training workflow** for an academic research assistant. The application uses Qwen2.5-0.5B-Instruct to turn informal topics into academic search queries, retrieves **actual indexed papers** from the Semantic Scholar Academic Graph API, and locally reranks candidates to present relevant results and source links.

> **Research result:** Fine-tuning made query rewrites more academic, but **mean top-five retrieved-paper relevance decreased from 4.68/5 (base) to 3.72/5 (fine-tuned)** on the five evaluated queries. This experiment demonstrates why stronger-looking generations are not necessarily better retrieval queries. The evaluation is limited to a small, manually rated set, so these results are not a general benchmark.

## Technology

Python · Qwen2.5-0.5B-Instruct · Hugging Face Transformers · PyTorch · LoRA / PEFT · TRL · Google Colab (Tesla T4) · FastAPI · Semantic Scholar API.

## Architecture

```text
User research topic
       |
       v
Qwen base model or Qwen + LoRA adapter
       |  academic search terms
       v
Semantic Scholar Academic Graph API
       |  fetch ~20 real paper candidates for a five-paper request
       v
Local, explainable ranker
       |  relevance + original result order + citation count + recency
       v
Top-ranked paper metadata, direct URLs and short result summary
```

The model **does not generate paper titles, citations, or URLs**. Semantic Scholar supplies indexed paper metadata; the ranker sorts those results.

## Contents

```text
app/
  main.py                  FastAPI endpoints (/health, /research); model-mode selector
  query_rewriter.py        Load the base Qwen model plus the research LoRA adapter
  query_rewriter_base.py   Baseline Qwen query rewriter (no adapter needed)
  semantic_scholar.py      API client; credentials loaded from .env
  ranker.py                Transparent local paper ranking
training/
  generate_dataset.py      Reproducible, template-based dataset generator
  dataset/train.jsonl      500 query-rewriting instruction/output pairs
  finetune_lora_colab.ipynb Original Qwen 0.5B LoRA training workflow (outputs removed)
reports/
  research_report.pdf      Research paper with methodology, findings and limitations
requirements.txt
.env.example
.gitignore
```

The earlier `finetune_lora_cpu.py` is **not included** because that local file actually configured a separate Qwen2.5-3B-Instruct QLoRA experiment, not the 0.5B LoRA experiment used in the report. **The Colab notebook is the source of truth for the reported training run.**

## Quick start: baseline (no trained adapter required)

The application was developed on an Intel Mac using CPU inference, with model training performed separately in Colab. Python **3.11** is recommended to match the recorded environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Optionally edit .env to provide a Semantic Scholar API key.
uvicorn app.main:app --reload
```

In another terminal:

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"query":"sleep academic performance college","limit":5}'
```

**Default:** `RESEARCH_ASSISTANT_MODEL=base` runs the original pretrained Qwen model and does not require a saved adapter. This mode illustrates the pipeline, **not** the fine-tuned research system.

The first run downloads the public base model. The Semantic Scholar API can impose rate limits; authenticated requests are supported by the optional `SEMANTIC_SCHOLAR_API_KEY` value in `.env`.

### Running with the fine-tuned research adapter

The learned LoRA adapter is **not published in this repository**. If you already have the locally saved adapter and its configuration, place the files under `research-assistant-model/` and set in `.env`:

```dotenv
RESEARCH_ASSISTANT_MODEL=finetuned
```

Then start the API as above. The application loads `Qwen/Qwen2.5-0.5B-Instruct` and attaches that adapter using PEFT. Without the adapter files, fine-tuned mode cannot start. Model weights, local caches, training checkpoints, and credentials are intentionally excluded from source control.

## Reproducing the dataset and GPU fine-tuning

The generator has **50 research topic families** with six general and four intent-specific phrasings per topic (recent studies, systematic reviews, foundational research, and research gaps). The data generator is deterministic with seed 42 and requires **no API key**.

```bash
python training/generate_dataset.py
```

In [Google Colab](https://colab.research.google.com/), select a GPU runtime and open `training/finetune_lora_colab.ipynb`. The notebook prompts you to upload `training/dataset/train.jsonl`. Follow the notebook cells to split data into training and validation examples, load the Qwen base model, apply LoRA, train, and export the adapter. The recorded experiment used a Tesla T4. Colab library versions and runtime availability may differ, so reruns might require compatibility adjustments; **the notebook has not been rerun as part of this repository preparation**.

| Recorded training configuration | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Dataset | 500 examples / 50 topic families |
| Split | 450 train / 50 validation (seed 42) |
| Training | 3 epochs; FP16; NVIDIA Tesla T4 |
| LoRA | rank 16; alpha 32; dropout 0.05 |
| Adapted modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable parameters | 2,162,688 (~0.4359% of base model) |
| Final recorded validation loss | 0.1137 |

## Evaluation summary

The five-query pilot evaluated query rewriting, top-five paper relevance, and end-to-end latency. These figures are measurements of **this particular dataset and system**, not general results for Qwen or LoRA.

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Average query-rewrite length | 4.8 words | 10.0 words |
| Original query-term preservation | 84.3% | 74.3% |
| Mean top-five paper relevance | **4.68/5** | **3.72/5** |
| Observed end-to-end response time | 3.967 s | 3.235 s |

The response times include Semantic Scholar API/network latency and local ranking, so they **do not show that LoRA accelerated model inference**. Paper relevance was rated manually for only five fixed queries by one non-blinded evaluator. See the full report for the evaluation rubric, cases, and reproducibility limits.

**Observed failure modes:** Semantic drift, loss of required population/intent terms, and insertion of unrequested constraints. A possible next iteration could combine original-query retrieval with rewritten-query retrieval and add relevance-aware evaluation before selecting a rewrite.

## Project scope and security

- The API provides paper discovery support, **not** validated citations or automatic assessment of scientific quality.
- An API key, if configured, belongs in `.env`, which is ignored by Git; commit `.env.example` only.
- Local model caches, exported adapter files, and notebooks' runtime outputs are excluded from this public source package.
- The dataset contains synthetic academic-query examples from a predefined topic/template generator.

## Research report

[Read the research report](reports/research_report.pdf) for the system design, training decisions, evaluation methodology, figures, and limitations.

**Author:** Tanmay Agarwal · Rider University · MacMillan Summer Fellowship (2026).
