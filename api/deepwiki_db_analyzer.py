#!/usr/bin/env python
"""
deepwiki_db_analyzer.py
========================
Loads a DeepWiki database pickle and prints a detailed summary.
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import adalflow.core.db.LocalDB
from adalflow.core.db import LocalDB
from adalflow.core.types import Document

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def get_default_root() -> Path:
    return Path.home() / ".adalflow"


def get_db_path(repo_name: str) -> Path:
    return get_default_root() / "databases" / f"{repo_name}.pkl"


def load_documents(db_path: Path) -> List[Document]:
    """Return the list of Document objects stored in the pickle."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    with db_path.open("rb") as f:
        # Load the LocalDB object
        db = pickle.load(f)

    if not isinstance(db, LocalDB):
        raise TypeError("Pickle does not contain a LocalDB object")

    # Get the transformed documents from the LocalDB instance
    # The key "split_and_embed" is used by DatabaseManager in api/data_pipeline.py
    documents = db.get_transformed_data(key="split_and_embed")

    if not isinstance(documents, list):
        raise TypeError("Transformed data from LocalDB is not a list of Documents")

    # Basic sanity check
    for i, doc in enumerate(documents):
        # Assuming adalflow.core.types.Document objects
        if not hasattr(doc, "text") or not hasattr(doc, "meta_data") or not hasattr(doc, "vector"):
            raise ValueError(f"Malformed Document at index {i}")

    return documents


def preview_text(text: str, width: int = 2, lines: int = 5) -> str:
    """Return a truncated preview of a long string."""
    wrapped = text.splitlines()
    preview = "\n".join(wrapped[:lines])
    return preview + ("…" if len(wrapped) > lines else "")

# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def report(docs: List[Document], json_output: bool = False, max_docs: int | None = None) -> str | None:
    docs = docs[:max_docs] if max_docs else docs

    if json_output:
        out = []
        for d in docs:
            meta = d.meta_data # Changed from d.metadata to d.meta_data
            out.append({
                "file_path": meta.get("file_path"),
                "chunk_id": meta.get("chunk_id"),
                "repo": meta.get("repo"),
                "owner": meta.get("owner"),
                "language": meta.get("language"),
                "source": meta.get("source"),
                "source_id": meta.get("source_id"),
                "content_length": len(d.text), # Changed from d.page_content to d.text
                "content_preview": preview_text(d.text), # Changed from d.page_content to d.text
            })
        return json.dumps(out, indent=4, ensure_ascii=False)

    # Human‑readable
    for i, d in enumerate(docs, start=1):
        meta = d.meta_data or {} # Changed from d.metadata to d.meta_data
        print(f"\n=== Document {i} ===")
        print(f"  file_path : {meta.get('file_path', 'N/A')}")
        print(f"  chunk_id  : {meta.get('chunk_id', 'N/A')}")
        print(f"  repo      : {meta.get('repo', 'N/A')}")
        print(f"  owner     : {meta.get('owner', 'N/A')}")
        print(f"  language  : {meta.get('language', 'N/A')}")
        print(f"  source    : {meta.get('source', 'N/A')}")
        print(f"  source_id : {meta.get('source_id', 'N/A')}")
        print(f"  content_length : {len(d.text)} bytes") # Changed from d.page_content to d.text
        print(f"  content_preview :\n{preview_text(d.text)}") # Changed from d.page_content to d.text

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="DeepWiki DB analyzer")
    parser.add_argument(
        "repo_name", help="owner_repo (e.g. man_deepwiki-open-)")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of documents to display")
    parser.add_argument("--json", action="store_true",
                        help="Print JSON instead of plain text")
    args = parser.parse_args()

    db_path = get_db_path(args.repo_name)
    try:
        docs = load_documents(db_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit()

    print(report(docs, json_output=args.json, max_docs=args.max))


if __name__ == "__main__":
    main()
