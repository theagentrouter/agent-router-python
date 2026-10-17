"""ADK function tools that work with actual KB passages and report files."""

from pathlib import Path

from .kb import KnowledgeBase


class WorkspaceTools:
    def __init__(self, kb: KnowledgeBase, output_directory: Path):
        self.kb = kb
        self.output_directory = output_directory.resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def search_knowledge(self, query: str, limit: int = 5) -> dict:
        """Search local policy documents. Returns passages with source names and L-prefixed lines.

        Args:
            query: Keywords to search for, such as annual refund or data export.
            limit: Number of passages, between 1 and 10.
        """
        try:
            return {"ok": True, "passages": self.kb.search(query, limit)}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    def read_document(self, source: str, start_line: int = 1, end_line: int = 0) -> dict:
        """Read up to 80 lines from a KB document returned by search_knowledge.

        Args:
            source: Exact document filename returned by search_knowledge.
            start_line: First line to read, counting from 1.
            end_line: Last line to read; 0 selects up to 80 lines automatically.
        """
        try:
            return {"ok": True, "passage": self.kb.read(source, start_line, end_line or None)}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    def save_report(self, filename: str, content: str) -> dict:
        """Save a cited report. Validates sources and line ranges and never overwrites a file.

        Args:
            filename: Plain .md, .txt, or .json filename; no directories.
            content: Report text, including citations such as [billing.md:L7-L9].
        """
        try:
            path = self.output_directory / filename
            if Path(filename).name != filename or path.suffix.lower() not in {
                ".md",
                ".txt",
                ".json",
            }:
                raise ValueError("Use a plain .md, .txt, or .json filename without directories")
            if not content or len(content) > 100_000:
                raise ValueError("Report must contain between 1 and 100000 characters")
            citations = self.kb.validate_citations(content)
            for suffix in range(1, 1001):
                candidate = path if suffix == 1 else path.with_stem(f"{path.stem}-{suffix}")
                if not candidate.resolve().is_relative_to(self.output_directory):
                    raise ValueError("Report must remain inside the output directory")
                try:
                    with candidate.open("x", encoding="utf-8") as handle:
                        handle.write(content)
                except FileExistsError:
                    continue
                return {"ok": True, "saved_to": str(candidate), "valid_citations": citations}
            raise ValueError("Choose another filename; too many existing reports")
        except (ValueError, OSError) as exc:
            return {"ok": False, "error": str(exc)}
