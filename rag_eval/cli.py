from __future__ import annotations
import argparse
from rouge_score import rouge_scorer

from .config import Settings
from .data import build_examples
from .openai_client import Chat
from .runner import run_one_example_on_model, summarize, write_csvs, timestamped_outdir

try:
    from bert_score import BERTScorer  # type: ignore
except Exception:
    BERTScorer = None  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FluentPathRAG vs concatenation on long-form answers.")
    parser.add_argument("--models", nargs="*", default=None, help="Override models, e.g. --models gpt-4.1 gpt-4o")
    parser.add_argument("--no-bertscore", action="store_true", help="Disable BERTScore")
    args = parser.parse_args()

    settings = Settings.from_env()
    model_list = [m.strip() for m in (args.models or settings.model_list) if m.strip()]

    # Metrics
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    berts = None
    if not args.no_bertscore and BERTScorer is not None:
        try:
            berts = BERTScorer(model_type=settings.bertscore_model, lang="en", rescale_with_baseline=True)
        except Exception as e:
            print(f"[WARN] BERTScore disabled ({type(e).__name__}: {e})")
            berts = None

    chat = Chat(api_key=settings.openai_api_key)

    all_rows = []
    exs = build_examples()

    print("=" * 96)
    print("Running models:", ", ".join(model_list))
    print("=" * 96)
    for i_model, model in enumerate(model_list, 1):
        print(f"\n## Model {i_model}/{len(model_list)}: {model}")
        for i, ex in enumerate(exs, 1):
            print(f"  - Example {i}/{len(exs)}: {ex.query}")
            rows = run_one_example_on_model(ex, model, rouge, berts, chat, settings)
            all_rows.extend(rows)
            for r in rows:
                print(f"    [{r['mode'].upper()}] RL={r['rougeL']} BLEU={r['bleu']} BERT-F1={r['bertscore_f1']}")

    summaries = summarize(all_rows)
    out_dir = timestamped_outdir(settings.out_dir)
    write_csvs(all_rows, summaries, out_dir)

    print("\n== SUMMARY (mean metrics) ==")
    for r in summaries:
        print(
            f"{r['model']:>10s} | {r['mode']:>6s} | N={r['n_examples']}" \
            f" | R1={r['mean_rouge1']} | RL={r['mean_rougeL']} | BLEU={r['mean_bleu']} | BERT-F1={r['mean_bertscore_f1']}"
        )
    print(f"\nResults written to: {out_dir}")