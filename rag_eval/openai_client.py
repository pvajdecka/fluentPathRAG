from typing import Dict, List, Optional
from openai import OpenAI


class Chat:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)


def call(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.0, max_tokens: int = 300, seed: Optional[int] = None) -> str:
    try:
        resp = self.client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {type(e).__name__}: {e}]"