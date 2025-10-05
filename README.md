# FluentPathRAG vs PathRAG-style Concatenation (Long‑form, Multi‑model)

This repo evaluates two **context-formatting** strategies for RAG over graph paths while **holding retrieval fixed**:

1. **PathRAG-style baseline** — linearize each path and **concatenate** them.
2. **FluentPathRAG** — rewrite *all* paths into a single, **query‑focused, fluent** evidence summary **before** answering.

Answers are constrained to **one sentence** (default **15–40 words**). We report **ROUGE‑1/ROUGE‑L (F1)**, **sacreBLEU**, and **BERTScore (F1)** per example and as per‑(model,mode) means.

## Examples included

1. Mercury — multi‑attribute traits
2. European colonization origins — causal chain
3. Smoking → DNA → malignancy → mortality — mechanism
4. Neural networks training — pipeline
5. Beethoven → Germany → Berlin → Spree — location chain

## Install & run

```bash
pip install -r requirements.txt
cp .env.example .env  # set your OPENAI_API_KEY
python scripts/run_eval.py --models gpt-4.1 gpt-4o
````

Optionally disable BERTScore (if PyTorch unavailable):

```bash
python scripts/run_eval.py --no-bertscore
```

Outputs are written under `outputs/<YYYYMMDD-HHMMSS>/`:

* `results_longform_fluent_vs_concat_detailed.csv`
* `results_longform_fluent_vs_concat_summary.csv`

## Configuration

Use `.env` (or environment variables) to tweak:

* `MODEL_LIST` — models to evaluate
* `MODEL_SUMMARIZER_FIXED` — force a fixed summarizer
* `ANSWER_MIN_WORDS` / `ANSWER_MAX_WORDS` — answer length range
* `BERTSCORE_MODEL` — BERTScore backbone (default `roberta-large`)
* `STRICT_CONTEXT_ONLY` — forbid outside knowledge (default on)
* `SEED` — seed where supported

## Notes

* Determinism is not guaranteed across API runs; we set `temperature=0.0` and pass a fixed `seed` where supported.
* API errors are captured in-line in the output text for post‑mortem.
* Secrets are **not** hard‑coded; use environment variables.

```
```
