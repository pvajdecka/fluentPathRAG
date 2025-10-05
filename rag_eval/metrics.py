from typing import Dict, List, Optional
from rouge_score import rouge_scorer
import sacrebleu


try:
    from bert_score import BERTScorer
except Exception:
    BERTScorer = None


def bleu_multi_ref(hyp: str, refs: List[str]) -> float:
    try:
        return sacrebleu.sentence_bleu(hyp, refs).score / 100
    except Exception:
        return 0.0


def rouge_best(hyp: str, refs: List[str], scorer) -> Dict[str, float]:
    best = {"rouge1": 0.0, "rougeL": 0.0}
    for ref in refs:
        s = scorer.score(ref, hyp)
        best["rouge1"] = max(best["rouge1"], s["rouge1"].fmeasure)
        best["rougeL"] = max(best["rougeL"], s["rougeL"].fmeasure)
    return best


def bertscore_best(hyp: str, refs: List[str], scorer: Optional[BERTScorer]) -> float:
    if scorer is None:
        return 0.0
    best = 0.0
    for ref in refs:
        _, _, F1 = scorer.score([hyp], [ref], device="cpu")
        best = max(best, float(F1[0]))
    return best