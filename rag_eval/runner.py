from __future__ import annotations
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple
from rouge_score import rouge_scorer

from .config import Settings
from .data import Example
from .formatters import make_concat_context, linearize_path
from .metrics import bleu_multi_ref, rouge_best, bertscore_best
from .openai_client import Chat

try:
    from bert_score import BERTScorer  # type: ignore
except Exception:
    BERTScorer = None  # type: ignore


def make_fluent_context(query: str, paths, summarizer_model: str, chat: Chat, min_words: int, max_words: int, seed: int) -> str:
    """Ask a model to weave paths into a faithful, answer-ready single-sentence context."""
    desc = "\n".join(f"- {linearize_path(p)}" for p in paths)
    target_low = max(min_words, max_words - 8)

    sys = (
        "You are an Evidence Weaver for multi-hop QA.\n"
        "You transform graph-path evidence into an answer-ready context while remaining 100% faithful.\n"
        "Optimization goal: maximize lexical overlap with relevant evidence (helps ROUGE) while avoiding any invented facts."
    )

    user = f"""
Question: {query}

Evidence chains (paths), formatted as [NodeID] text --relation→ [NodeID] text:
{desc}

TASK (follow strictly)
1) Must-keep phrases (copy verbatim; 6–12 short spans): names, numerals/dates, places, technical terms.
2) Facts: concise bullets using original surface forms when possible.
3) Synthesis (1 sentence): order facts so the answer is obvious using explicit connectors.
4) FINAL: On the LAST line, write ONE standalone English sentence of {min_words}–{max_words} words (aim for {target_low}–{max_words}). No quotes/lists/meta.
If evidence is insufficient, output exactly: Insufficient context.

FORMAT (exactly; last line must be only the final sentence):
Must-keep phrases:
- ...
- ...
Facts:
- ...
- ...
Synthesis: <one sentence here>
<final answer sentence here>
""".strip()

    return chat.call(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model=summarizer_model,
        temperature=0.0,
        max_tokens=360,
        seed=seed,
    )


def answer_with_context(query: str, context: str, answer_model: str, chat: Chat, min_words: int, max_words: int, strict_only: bool, seed: int) -> str:
    """Answer using ONLY the provided context as a single {min_words}–{max_words} word sentence."""
    sys_core = "You are a careful QA assistant."
    if strict_only:
        sys_core += " Use ONLY the facts in the supplied context. If insufficient, respond exactly: Insufficient context."

    user = f"""
Context:
\"\"\"{context}\"\"\"

Question: {query}

Instructions:
- Write EXACTLY ONE well-formed English sentence between {min_words} and {max_words} words.
- No lists/bullets; no quotations; declarative tone.
- If the answer isn't derivable from the context, reply exactly: Insufficient context.
""".strip()

    return chat.call(
        messages=[{"role": "system", "content": sys_core}, {"role": "user", "content": user}],
        model=answer_model,
        temperature=0.0,
        max_tokens=160,
        seed=seed,
    )


def run_one_example_on_model(
    ex: Example,
    model_name: str,
    rouge: rouge_scorer.RougeScorer,
    berts,
    chat: Chat,
    settings: Settings,
) -> List[Dict[str, str]]:
    """Run concat + fluent for one example and score both hypotheses."""
    # Build contexts
    ctx_concat = make_concat_context(ex.paths)
    summarizer_model = settings.model_summarizer_fixed or model_name
    ctx_fluent = make_fluent_context(ex.query, ex.paths, summarizer_model, chat, settings.min_words, settings.max_words, settings.seed)

    # Answers
    ans_concat = answer_with_context(ex.query, ctx_concat, model_name, chat, settings.min_words, settings.max_words, settings.strict_context_only, settings.seed)
    ans_fluent = answer_with_context(ex.query, ctx_fluent, model_name, chat, settings.min_words, settings.max_words, settings.strict_context_only, settings.seed)

    # Metrics
    r_concat = rouge_best(ans_concat, ex.gold_refs, rouge)
    bleu_concat = bleu_multi_ref(ans_concat, ex.gold_refs)
    bsf_concat = bertscore_best(ans_concat, ex.gold_refs, berts)

    r_fluent = rouge_best(ans_fluent, ex.gold_refs, rouge)
    bleu_fluent = bleu_multi_ref(ans_fluent, ex.gold_refs)
    bsf_fluent = bertscore_best(ans_fluent, ex.gold_refs, berts)

    return [
        {
            "model": model_name,
            "mode": "concat",
            "query": ex.query,
            "answer": ans_concat,
            "rouge1": f"{r_concat['rouge1']:.4f}",
            "rougeL": f"{r_concat['rougeL']:.4f}",
            "bleu": f"{bleu_concat:.4f}",
            "bertscore_f1": f"{bsf_concat:.4f}",
        },
        {
            "model": model_name,
            "mode": "fluent",
            "query": ex.query,
            "answer": ans_fluent,
            "rouge1": f"{r_fluent['rouge1']:.4f}",
            "rougeL": f"{r_fluent['rougeL']:.4f}",
            "bleu": f"{bleu_fluent:.4f}",
            "bertscore_f1": f"{bsf_fluent:.4f}",
        },
    ]


def summarize(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Aggregate mean metrics per (model, mode)."""
    # group by (model, mode)
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for r in rows:
        key = (r["model"], r["mode"])
        groups.setdefault(key, []).append(r)

    summary_rows: List[Dict[str, str]] = []
    for (model, mode), items in groups.items():
        n = len(items)
        mean_r1 = sum(float(x["rouge1"]) for x in items) / n
        mean_rl = sum(float(x["rougeL"]) for x in items) / n
        mean_bleu = sum(float(x["bleu"]) for x in items) / n
        mean_bsf = sum(float(x["bertscore_f1"]) for x in items) / n
        summary_rows.append({
            "model": model,
            "mode": mode,
            "n_examples": str(n),
            "mean_rouge1": f"{mean_r1:.4f}",
            "mean_rougeL": f"{mean_rl:.4f}",
            "mean_bleu": f"{mean_bleu:.4f}",
            "mean_bertscore_f1": f"{mean_bsf:.4f}",
        })
    return summary_rows


def write_csvs(rows: List[Dict[str, str]], summaries: List[Dict[str, str]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    detail = os.path.join(out_dir, "results_longform_fluent_vs_concat_detailed.csv")
    summary = os.path.join(out_dir, "results_longform_fluent_vs_concat_summary.csv")

    with open(detail, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "mode", "query", "answer", "rouge1", "rougeL", "bleu", "bertscore_f1"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "mode", "n_examples", "mean_rouge1", "mean_rougeL", "mean_bleu", "mean_bertscore_f1"],
        )
        writer.writeheader()
        writer.writerows(summaries)


def timestamped_outdir(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(base, ts)