from __future__ import annotations
from typing import List
from .data import Path

def linearize_path(path: Path) -> str:
    """Render a path as a labeled sequence: [N1] text --rel→ [N2] text ..."""
    parts = []
    for i, node in enumerate(path.nodes):
        parts.append(f"[{node.id}] {node.text}")
        if i < len(path.edges):
            e = path.edges[i]
            # light safety check on alignment
            if e.src != path.nodes[i].id or e.dst != path.nodes[i + 1].id:
                parts.append("  (⚠ edge-node mismatch)  ")
            parts.append(f"  --{e.relation}→  ")
    return "".join(parts)


def make_concat_context(paths: List[Path]) -> str:
    """Baseline: list each linearized path, separated by blank lines."""
    return "\n\n".join(linearize_path(p) for p in paths)