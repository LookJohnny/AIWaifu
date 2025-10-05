#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown -> JSON converter for RAG data

Features
- Supports Markdown files with or without YAML-like front matter
- If front matter exists, it maps to id/title/tags/meta; body -> text
- If no front matter, splits by '## ' headings into multiple records
- Simple, dependency-free parsing (no PyYAML required)

Usage
  py scripts/md_to_json.py --input docs/ --output rag/data/converted.json --store-id 001 --lang zh-CN

Notes
- Required fields per record: title, text
- Optional: id, tags(list), meta(dict)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


Record = Dict[str, object]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_front_matter(md: str) -> Tuple[Optional[Dict[str, object]], str]:
    """Parse minimal YAML-like front matter.

    Supports forms like:
    ---
    id: sku-1001
    title: 夏季新品清凉饮品
    tags: [饮品, 促销, 夏季]
    meta: {store_id: "001", price: 18.0}
    ---
    正文...
    """
    if not md.startswith("---\n"):
        return None, md

    end = md.find("\n---\n", 4)
    if end == -1:
        # support Windows line endings too
        end = md.find("\r\n---\r\n", 4)
        if end == -1:
            return None, md

    header = md[4:end].strip("\r\n")
    body = md[end + (5 if md[end:end+5] == "\n---\n" else 7):]

    data: Dict[str, object] = {}
    lines = header.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith('#'):
            continue
        # key: value
        m = re.match(r"^([A-Za-z0-9_]+)\s*:\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        # list inline: [a, b]
        if val.startswith('[') and val.endswith(']'):
            items = [x.strip().strip('"\'') for x in val[1:-1].split(',') if x.strip()]
            data[key] = items
            continue
        # dict inline: {a: 1, b: 2} (treat like JSON-ish)
        if val.startswith('{') and val.endswith('}'):
            try:
                # Replace YAML-like key: value with JSON-like "key": value
                j = re.sub(r"([A-Za-z0-9_]+)\s*:", r'"\1":', val)
                data[key] = json.loads(j)
            except Exception:
                data[key] = {"raw": val}
            continue
        # multiline list starting with '-'
        if val == '' and i < len(lines) and lines[i].lstrip().startswith('-'):
            items: List[str] = []
            while i < len(lines) and lines[i].lstrip().startswith('-'):
                items.append(lines[i].lstrip()[1:].strip())
                i += 1
            data[key] = items
            continue
        # plain string (allow tags: a,b,c)
        if key == 'tags':
            parts = [p.strip() for p in re.split(r"[|,]", val) if p.strip()]
            data[key] = parts
        else:
            data[key] = val

    return data, body


def split_sections_by_h2(md: str) -> List[Tuple[str, str]]:
    """Split by '## ' headings; return list of (title, text)."""
    # Normalize newlines
    text = md.replace('\r\n', '\n')
    parts = re.split(r"\n(?=##\s+)", text)
    sections: List[Tuple[str, str]] = []
    for part in parts:
        if not part.strip():
            continue
        if part.startswith('## '):
            first_nl = part.find('\n')
            title = part[3:first_nl].strip() if first_nl != -1 else part[3:].strip()
            body = part[first_nl+1:] if first_nl != -1 else ''
        else:
            # no heading at very beginning; use file intro as single section (title empty)
            title = ''
            body = part
        sections.append((title, body.strip()))
    return sections


def records_from_markdown(path: Path, id_prefix: str = '') -> List[Record]:
    md = read_text(path)
    fm, body = parse_front_matter(md)
    recs: List[Record] = []
    if fm:
        title = str(fm.get('title') or '').strip()
        text = body.strip()
        if not title:
            # fallback: first heading or first 40 chars
            secs = split_sections_by_h2(body)
            if secs:
                title = secs[0][0] or (secs[0][1][:40] + '...')
        if text:
            recs.append({
                'id': fm.get('id') or f"md:{path.stem}",
                'title': title,
                'text': text,
                'tags': fm.get('tags') or [],
                'meta': fm.get('meta') or {},
                'source': str(path)
            })
        return recs

    # no front matter: split by H2
    secs = split_sections_by_h2(md)
    if not secs:
        # single record fallback
        title = path.stem
        recs.append({
            'id': f"md:{path.stem}",
            'title': title,
            'text': md.strip(),
            'tags': [],
            'meta': {},
            'source': str(path)
        })
        return recs

    for idx, (title, text) in enumerate(secs, 1):
        if not text.strip():
            continue
        t = title if title else (text[:40] + '...' if len(text) > 43 else text)
        recs.append({
            'id': f"md:{path.stem}:{idx}" if not id_prefix else f"{id_prefix}:{path.stem}:{idx}",
            'title': t.strip(),
            'text': text.strip(),
            'tags': [],
            'meta': {},
            'source': str(path)
        })
    return recs


def validate_records(records: List[Record]) -> List[str]:
    errors: List[str] = []
    seen = set()
    for i, r in enumerate(records):
        rid = str(r.get('id') or f'#{i}')
        title = (r.get('title') or '').strip()
        text = (r.get('text') or '').strip()
        if not title:
            errors.append(f"{rid}: missing title")
        if not text:
            errors.append(f"{rid}: missing text")
        if rid in seen:
            errors.append(f"{rid}: duplicated id")
        seen.add(rid)
        # meta must be dict
        meta = r.get('meta')
        if meta is not None and not isinstance(meta, dict):
            errors.append(f"{rid}: meta must be object")
        # tags must be list
        tags = r.get('tags')
        if tags is not None and not isinstance(tags, list):
            errors.append(f"{rid}: tags must be array")
    return errors


def collect_markdown(input_path: Path) -> List[Path]:
    paths: List[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() in {'.md', '.markdown'}:
            paths.append(input_path)
    else:
        for p in input_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in {'.md', '.markdown'}:
                paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Convert Markdown to RAG JSON records")
    ap.add_argument('--input', '-i', required=True, help='Markdown file or directory')
    ap.add_argument('--output', '-o', help='Output JSON file (default: rag/data/converted_<name>.json)')
    ap.add_argument('--store-id', help='Set meta.store_id for all records')
    ap.add_argument('--lang', help='Set meta.lang for all records')
    ap.add_argument('--id-prefix', default='', help='Prefix for generated ids')
    ap.add_argument('--dry-run', action='store_true', help='Do not write file, only print stats')
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    md_files = collect_markdown(input_path)
    if not md_files:
        raise SystemExit("No markdown files found")

    all_records: List[Record] = []
    for p in md_files:
        recs = records_from_markdown(p, id_prefix=args.id_prefix)
        # enrich meta
        for r in recs:
            meta = dict(r.get('meta') or {})
            if args.store_id:
                meta.setdefault('store_id', args.store_id)
            if args.lang:
                meta.setdefault('lang', args.lang)
            r['meta'] = meta
        all_records.extend(recs)

    errors = validate_records(all_records)
    print(f"Parsed {len(md_files)} markdown files -> {len(all_records)} records")
    if errors:
        print("Validation errors:")
        for e in errors:
            print(" -", e)
        # keep writing unless dry-run; allow user to fix later

    if args.dry_run:
        # Print first 2 samples
        for s in all_records[:2]:
            print(json.dumps(s, ensure_ascii=False)[:400] + ('...' if len(json.dumps(s, ensure_ascii=False))>400 else ''))
        return

    # determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        base = input_path.stem if input_path.is_file() else input_path.name
        out_path = Path('rag/data') / f'converted_{base}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote: {out_path}")


if __name__ == '__main__':
    main()

