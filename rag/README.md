RAG Folder

Overview
- This folder hosts Retrieval-Augmented Generation (RAG) assets and helpers.
- Designed for incremental growth: drop data under `rag/data/`, optional indexes under `rag/index/`, and use `rag/knowledge.py` to load/search.

Structure
- `rag/data/`        Raw knowledge files (JSON/CSV/MD exported to JSON)
- `rag/index/`       Future vector indexes or caches (kept empty for now)
- `rag/knowledge.py` Minimal retriever that loads and searches local data

Data formats
1) JSON (preferred)
   - Array of objects with fields:
     - `id` (optional)
     - `title` (string)
     - `text` (string) – the main content
     - `tags` (optional array of strings)
     - `meta` (optional object)

2) CSV
   - Columns: `title`, `text`, optional `tags` (comma-separated), `id`, `meta` (JSON)

Usage (Python)
```python
from rag.knowledge import KnowledgeBase

kb = KnowledgeBase()  # loads from rag/data by default
results = kb.search("优惠活动怎么参加", top_k=3)
for r in results:
    print(r["score"], r["title"])
```

Convert Markdown to JSON (CLI)
```bash
# Convert a single markdown file into JSON under rag/data/
py scripts/md_to_json.py --input docs/promo.md --output rag/data/promo.json --store-id 001 --lang zh-CN

# Convert an entire folder of markdown files, auto naming output
py scripts/md_to_json.py --input docs/ --id-prefix md --store-id 001
```

Notes
- The current search is a lightweight lexical scorer (English tokens + Chinese bigrams). It’s dependency‑free and good enough to bootstrap. You can swap in a vector index later without changing callers.
- Keep large binary files (images/videos) outside of `data/`; store links in `meta`.
