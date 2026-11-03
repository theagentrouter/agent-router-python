# SPDX-License-Identifier: Apache-2.0
"""Write local-gateway/THIRD_PARTY_NOTICES.txt: the licences of the Go code compiled into aigw.

Needs Go. It downloads the pinned aigw program for every platform, because each build links a slightly
different set of modules, and refuses a checksum mismatch. Run it when aigw changes: the pull request
update-aigw.yml opens lists this file among those that still name the old version.

    python scripts/aigw_notices.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "local-gateway" / "aigw-release.json"
OUT = ROOT / "local-gateway" / "THIRD_PARTY_NOTICES.txt"
LICENCE_FILE = re.compile(r"^(licen[cs]e|copying|notice|copyright|patents)([._-].*)?$", re.IGNORECASE)
NOT_A_LICENCE = (".go", ".py", ".sh", ".json", ".yaml", ".yml", ".tmpl", ".proto", ".c", ".h", ".s")
RULE = "=" * 80


def _run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def _download(url: str, expected: str, path: Path) -> None:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=300) as response, path.open("wb") as output:
        while block := response.read(1024 * 1024):
            digest.update(block)
            output.write(block)
    if digest.hexdigest() != expected:
        raise SystemExit(f"SHA-256 mismatch for {url}")


def _build_info(binary: Path) -> tuple[str, set[str]]:
    """The Go version that built ``binary``, and its modules as ``path@version``."""
    lines = _run("go", "version", "-m", str(binary)).splitlines()
    modules: list[str] = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) > 3 and fields[1] in ("mod", "dep"):
            modules.append(f"{fields[2]}@{fields[3]}")
        elif len(fields) > 3 and fields[1] == "=>":  # the module above is replaced by this one
            modules[-1] = f"{fields[2]}@{fields[3]}"
    return lines[0].rsplit(" ", 1)[-1], set(modules)


def _licence_texts(folder: Path) -> list[tuple[str, str]]:
    found = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and LICENCE_FILE.match(path.name) and path.suffix.lower() not in NOT_A_LICENCE:
            found.append((path.name, path.read_text(errors="replace").replace("\r\n", "\n").strip()))
    return found


def _module_folders(modules: set[str], work: Path) -> dict[str, Path]:
    done = subprocess.run(
        ["go", "mod", "download", "-json", *sorted(modules)], cwd=work, capture_output=True, text=True, check=False
    )
    decoder, text, position = json.JSONDecoder(), done.stdout, 0
    folders: dict[str, Path] = {}
    while position < len(text.rstrip()):
        entry, end = decoder.raw_decode(text, position)
        position = end + 1
        if entry.get("Error") or not entry.get("Dir"):
            raise SystemExit(f"cannot download {entry.get('Path')}@{entry.get('Version')}: {entry.get('Error')}")
        folders[f"{entry['Path']}@{entry['Version']}"] = Path(entry["Dir"])
    if set(folders) != modules:
        raise SystemExit(f"go mod download left out {sorted(modules - set(folders))}: {done.stderr[:500]}")
    return folders


def _texts_by_module(folders: dict[str, Path]) -> dict[str, list[tuple[str, str]]]:
    """Each module's licence files. A module without any takes those of the module whose path contains it."""
    own = {module: _licence_texts(folder) for module, folder in folders.items()}
    by_path = {module.rsplit("@", 1)[0]: texts for module, texts in own.items() if texts}
    missing = []
    for module, texts in own.items():
        if texts:
            continue
        parts = module.rsplit("@", 1)[0].split("/")
        parents = ("/".join(parts[:end]) for end in range(len(parts) - 1, 0, -1))
        parent = next((path for path in parents if path in by_path), None)
        if parent is None:
            missing.append(module)
        else:
            own[module] = by_path[parent]
    if missing:
        raise SystemExit("no licence file found for:\n  " + "\n  ".join(sorted(missing)))
    return own


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    version, repository = manifest["version"], manifest["repository"]
    go_versions: set[str] = set()
    modules: set[str] = set()
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        for target, expected in manifest["assets"].items():
            binary = work / f"aigw-{target}"
            _download(f"https://github.com/{repository}/releases/download/{version}/aigw-{target}", expected, binary)
            go_version, found = _build_info(binary)
            go_versions.add(go_version)
            modules |= found
            binary.unlink()
        texts = _texts_by_module(_module_folders(modules, work))
    goroot = Path(_run("go", "env", "GOROOT").strip())
    # Homebrew keeps it one folder above GOROOT.
    go_licence = goroot / "LICENSE" if (goroot / "LICENSE").is_file() else goroot.parent / "LICENSE"
    go_texts = [("LICENSE", go_licence.read_text().strip())]

    groups: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for module, files in texts.items():
        groups.setdefault(tuple(files), []).append(module.replace("@", " "))
    sections = [([f"Go, its standard library and runtime ({', '.join(sorted(go_versions))})"], go_texts)]
    sections += sorted(((sorted(names), list(files)) for files, names in groups.items()), key=lambda s: s[0])

    lines = [
        f"Third-party notices for aigw {version}",
        "",
        f"The aigw program in this package is built by Agent Router (https://github.com/{repository})",
        "from the Go code listed here. Each section names the modules and gives their licence files.",
    ]
    for names, files in sections:
        lines += ["", RULE, *names, RULE]
        for name, text in files:
            lines += ["", f"--- {name}", "", text]
    OUT.write_text("\n".join(lines) + "\n")

    kinds: dict[str, int] = {}
    for files, names in groups.items():
        body = " ".join(text for _, text in files).lower()
        for kind, mark in (
            ("Apache", "apache license"),
            ("MIT", "permission is hereby granted, free of charge"),
            ("BSD", "redistribution and use in source and binary forms"),
            ("ISC", "permission to use, copy, modify, and/or distribute"),
            ("MPL", "mozilla public license"),
            ("GPL or LGPL", "gnu general public license"),
            ("GPL or LGPL", "gnu lesser general public license"),
        ):
            if mark in body:
                kinds[kind] = kinds.get(kind, 0) + len(names)
    print(f"{len(modules)} modules in {len(groups)} groups, {OUT.stat().st_size // 1024} KB, built with {go_versions}")
    print("modules mentioning each licence:", dict(sorted(kinds.items())))


if __name__ == "__main__":
    main()
