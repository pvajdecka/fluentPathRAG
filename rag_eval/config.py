from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()  # load .env if present

@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model_list: List[str]
    model_summarizer_fixed: Optional[str]
    seed: int
    strict_context_only: bool
    bertscore_model: str
    min_words: int
    max_words: int
    out_dir: str = "outputs"

    @staticmethod
    def from_env() -> "Settings":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY. Set it in your environment or .env file.")
        models = [m.strip() for m in os.getenv("MODEL_LIST", "gpt-4.1,gpt-4o").split(",") if m.strip()]
        summarizer_fixed = os.getenv("MODEL_SUMMARIZER_FIXED", "").strip() or None
        seed = int(os.getenv("SEED", "42"))
        strict = os.getenv("STRICT_CONTEXT_ONLY", "1").strip() not in {"0", "false", "False"}
        berts_model = os.getenv("BERTSCORE_MODEL", "roberta-large").strip()
        min_w = int(os.getenv("ANSWER_MIN_WORDS", "15"))
        max_w = int(os.getenv("ANSWER_MAX_WORDS", "40"))
        return Settings(
            openai_api_key=key,
            model_list=models,
            model_summarizer_fixed=summarizer_fixed,
            seed=seed,
            strict_context_only=strict,
            bertscore_model=berts_model,
            min_words=min_w,
            max_words=max_w,
        )