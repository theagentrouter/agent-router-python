from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_checks_detect_companion_version_drift(tmp_path, monkeypatch):
    script = _script("check_release")
    for name in (
        "pyproject.toml",
        "local-gateway/pyproject.toml",
        "src/theagentrouter/__init__.py",
        "local-gateway/src/theagentrouter_local_gateway/__init__.py",
        "packages/theagentrouter/pyproject.toml",
        "CHANGELOG.md",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    script.ROOT = tmp_path
    monkeypatch.setattr(sys, "argv", ["check_release"])
    script.main()
    package = tmp_path / "local-gateway/pyproject.toml"
    package.write_text(package.read_text().replace('version = "', 'version = "9.', 1))
    with pytest.raises(SystemExit, match="Version mismatch"):
        script.main()


def test_release_checks_require_the_full_package_to_pin_the_sdk(tmp_path, monkeypatch):
    script = _script("check_release")
    for name in (
        "pyproject.toml",
        "local-gateway/pyproject.toml",
        "src/theagentrouter/__init__.py",
        "local-gateway/src/theagentrouter_local_gateway/__init__.py",
        "packages/theagentrouter/pyproject.toml",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    script.ROOT = tmp_path
    monkeypatch.setattr(sys, "argv", ["check_release"])
    full = tmp_path / "packages/theagentrouter/pyproject.toml"
    full.write_text(full.read_text().replace("theagentrouter-core==", "theagentrouter-core>="))
    with pytest.raises(SystemExit, match="does not pin theagentrouter-core"):
        script.main()


def test_release_checks_reject_wrong_tag(monkeypatch):
    script = _script("check_release")
    monkeypatch.setattr(sys, "argv", ["check_release", "--tag", "v9.9.9"])
    with pytest.raises(SystemExit, match="does not match"):
        script.main()


def test_release_sets_the_version_everywhere(tmp_path, monkeypatch):
    script = _script("release")
    for name in (*script.FILES, "CHANGELOG.md"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n\n- A change.\n")
    script.ROOT = tmp_path
    current = script.current_version()
    script.set_version("9.8.7a6", "2026-10-13")
    for name in script.FILES:
        text = (tmp_path / name).read_text()
        assert "9.8.7a6" in text and current not in text, name
    assert "## [9.8.7a6] - 2026-10-13\n\n- A change." in (tmp_path / "CHANGELOG.md").read_text()
    # The result passes the same check the release workflow runs on the tag.
    check = _script("check_release")
    check.ROOT = tmp_path
    monkeypatch.setattr(sys, "argv", ["check_release", "--tag", "v9.8.7a6"])
    check.main()
    with pytest.raises(SystemExit, match='no "## Unreleased"'):
        script.set_version("9.8.8", "2026-10-14")


def test_binary_download_rejects_bad_checksum_and_keeps_existing_binary(tmp_path, monkeypatch):
    script = _script("fetch_aigw")
    script.ROOT = tmp_path
    folder = tmp_path / "local-gateway/src/theagentrouter_local_gateway/bin"
    folder.mkdir(parents=True)
    binary = folder / "aigw"
    binary.write_bytes(b"previous binary")
    (tmp_path / "local-gateway/aigw-release.json").write_text(
        json.dumps(
            {
                "repository": "acme/router",
                "version": "v1",
                "assets": {"darwin-arm64": "0" * 64},
            }
        )
    )
    monkeypatch.setattr(script.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(script.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(script.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong binary"))
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        script.main()
    assert binary.read_bytes() == b"previous binary"
    assert list(folder.iterdir()) == [binary]
