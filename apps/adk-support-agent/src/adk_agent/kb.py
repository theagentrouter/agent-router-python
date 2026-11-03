# SPDX-License-Identifier: Apache-2.0
"""Local Markdown/text retrieval using SQLite FTS5 and BM25 ranking."""

import re
import sqlite3
from pathlib import Path


class KnowledgeBase:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve(strict=True)
        if not self.directory.is_dir():
            raise ValueError("Knowledge base must be a directory")
        self.documents: dict[str, list[str]] = {}
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5("
            "source UNINDEXED, start_line UNINDEXED, end_line UNINDEXED, text, "
            "tokenize='porter unicode61')"
        )
        self.refresh()

    def refresh(self) -> int:
        """Rebuild a snapshot so deleted or edited documents do not leave stale hits."""
        documents = {}
        for path in sorted(self.directory.rglob("*")):
            relative = path.relative_to(self.directory)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
                continue
            if not path.resolve().is_relative_to(self.directory) or path.is_symlink():
                continue
            if path.stat().st_size > 2_000_000:
                raise ValueError(f"KB document exceeds 2 MB: {relative}")
            lines = path.read_text(encoding="utf-8").splitlines()
            if any(len(line) > 8000 for line in lines):
                raise ValueError(f"KB document has a line longer than 8000 characters: {relative}")
            documents[relative.as_posix()] = lines
        with self.db:
            self.db.execute("DELETE FROM chunks")
            for source, lines in documents.items():
                # Small overlapping chunks keep context and exact citation line numbers.
                start = 0
                while start < len(lines):
                    end, size = start, 0
                    while end < len(lines) and end - start < 32:
                        length = len(lines[end]) + 1
                        if end > start and size + length > 8000:
                            break
                        size += length
                        end += 1
                    text = "\n".join(lines[start:end])
                    if text.strip():
                        self.db.execute(
                            "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                            (source, start + 1, end, text),
                        )
                    if end == len(lines):
                        break
                    start = max(start + 1, end - 8)
        self.documents = documents
        return len(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10")
        # Quote individual terms so model-provided FTS operators cannot alter the query.
        terms = list(dict.fromkeys(re.findall(r"\w+", query.lower())))[:24]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        rows = self.db.execute(
            "SELECT source, start_line, end_line, text FROM chunks "
            "WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
            (expression, limit),
        ).fetchall()
        hits = [dict(row) for row in rows]
        for hit in hits:
            hit["text"] = self.number_lines(hit["text"].splitlines(), hit["start_line"])
        return hits

    @staticmethod
    def number_lines(lines: list[str], start: int) -> str:
        return "\n".join(f"L{number}: {line}" for number, line in enumerate(lines, start))

    def validate_citations(self, content: str) -> int:
        citations = re.findall(r"\[([^\[\]\n]+):L(\d+)(?:-L?(\d+))?\]", content)
        if not citations:
            raise ValueError("Include a KB citation such as [onboarding.md:L7-L9]")
        for source, first, last in citations:
            start, end = int(first), int(last or first)
            if source not in self.documents:
                raise ValueError(f"Unknown citation source: {source}. Copy the exact KB filename.")
            lines = self.documents[source]
            if not 1 <= start <= end <= len(lines):
                raise ValueError(f"Invalid citation range: {source}:L{start}-L{end}")
            if not any(line.strip() for line in lines[start - 1 : end]):
                raise ValueError(f"Citation refers only to blank lines: {source}:L{start}-L{end}")
        return len(citations)

    def read(self, source: str, start_line: int = 1, end_line: int | None = None) -> dict:
        if source not in self.documents:
            raise ValueError("Unknown source; use a source returned by search_knowledge")
        lines = self.documents[source]
        end = min(end_line if end_line is not None else start_line + 79, len(lines))
        if start_line < 1 or end < start_line or end - start_line >= 80:
            raise ValueError("Read between 1 and 80 existing lines")
        text = "\n".join(lines[start_line - 1 : end])
        if len(text) > 20_000:
            raise ValueError("Passage exceeds 20000 characters; request fewer lines")
        return {
            "source": source,
            "start_line": start_line,
            "end_line": end,
            "text": self.number_lines(lines[start_line - 1 : end], start_line),
        }

    def close(self) -> None:
        self.db.close()
