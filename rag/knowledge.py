from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


Record = Dict[str, object]


def _is_cjk(ch: str) -> bool:
    return '\u4e00' <= ch <= '\u9fff'


def _extract_tokens(text: str) -> List[str]:
    """Return simple tokens for scoring (EN words + CJK bigrams)."""
    t = text or ""
    words = re.findall(r"[A-Za-z0-9]+", t.lower())
    cjks = [c for c in t if _is_cjk(c)]
    bigrams = [cjks[i] + cjks[i + 1] for i in range(len(cjks) - 1)] if len(cjks) > 1 else []
    return words + bigrams


def _score(query: str, text: str) -> float:
    if not query or not text:
        return 0.0
    q = query.strip()
    t = text
    # strong match if full query substring appears
    if q in t:
        return 100.0
    q_tokens = set(_extract_tokens(q))
    t_tokens = set(_extract_tokens(t))
    if not q_tokens or not t_tokens:
        return 0.0
    inter = q_tokens.intersection(t_tokens)
    return float(len(inter))


def _read_json(path: Path) -> List[Record]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    out: List[Record] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or row.get("content") or "").strip()
        if not title and text:
            title = (text[:40] + "...") if len(text) > 43 else text
        rec: Record = {
            "id": row.get("id", f"json:{path.name}:{i}"),
            "title": title,
            "text": text,
            "tags": row.get("tags", []),
            "meta": row.get("meta", {}),
            "source": str(path),
        }
        out.append(rec)
    return out


def _read_csv(path: Path) -> List[Record]:
    out: List[Record] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            title = (row.get("title") or "").strip()
            text = (row.get("text") or row.get("content") or "").strip()
            tags_raw = (row.get("tags") or "").strip()
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
            meta_raw = (row.get("meta") or "").strip()
            try:
                meta = json.loads(meta_raw) if meta_raw else {}
            except Exception:
                meta = {"raw": meta_raw}
            if not title and text:
                title = (text[:40] + "...") if len(text) > 43 else text
            out.append({
                "id": row.get("id", f"csv:{path.name}:{i}"),
                "title": title,
                "text": text,
                "tags": tags,
                "meta": meta,
                "source": str(path),
            })
    return out


@dataclass
class KnowledgeBase:
    root: Path = Path("rag/data")
    records: Optional[List[Record]] = None

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.records is None:
            self.records = self._load_all()

    def _load_all(self) -> List[Record]:
        items: List[Record] = []
        if not self.root.exists():
            return items
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            try:
                if path.suffix.lower() == ".json":
                    items.extend(_read_json(path))
                elif path.suffix.lower() == ".csv":
                    items.extend(_read_csv(path))
            except Exception as e:
                print(f"[RAG] Failed to read {path}: {e}")
        print(f"[RAG] Loaded {len(items)} records from {self.root}")
        return items

    def refresh(self):
        self.records = self._load_all()

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[Record]:
        recs = self.records or []
        scored = []
        for r in recs:
            s = _score(query, str(r.get("text", "")))
            if s >= min_score:
                scored.append((s, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: List[Record] = []
        for s, r in scored[:top_k]:
            rr = dict(r)
            rr["score"] = s
            # build snippet
            text = str(rr.get("text", ""))
            pos = text.find(query)
            if pos >= 0:
                start = max(0, pos - 30)
                end = min(len(text), pos + len(query) + 60)
                rr["snippet"] = text[start:end]
            else:
                rr["snippet"] = text[:90]
            results.append(rr)
        return results

