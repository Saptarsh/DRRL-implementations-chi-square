"""Atomic, provenance-rich artifacts for RVChi2-DQN runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import gymnasium
import numpy
import pandas
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_path(path: Path) -> tuple[Path, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return Path(name), descriptor


def write_json_atomic(path: Path | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    temporary, descriptor = _atomic_path(destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv_atomic(
    path: Path | str,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    destination = Path(path)
    materialized = list(rows)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    temporary, descriptor = _atomic_path(destination)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def torch_save_atomic(path: Path | str, payload: Any) -> None:
    destination = Path(path)
    temporary, descriptor = _atomic_path(destination)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_npz_atomic(path: Path | str, **arrays: Any) -> None:
    destination = Path(path)
    temporary, descriptor = _atomic_path(destination)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            numpy.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def git_provenance(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)

    def run(*arguments: str) -> str:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    status = run("status", "--porcelain=v1")
    return {
        "revision": run("rev-parse", "HEAD") or None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "pandas": pandas.__version__,
    }


def artifact_hashes(directory: Path | str, names: Iterable[str]) -> dict[str, str]:
    root = Path(directory)
    return {name: sha256_file(root / name) for name in names}


def prepare_new_directory(path: Path | str) -> Path:
    """Create a result directory without overwriting an earlier run."""

    destination = Path(path)
    if destination.exists():
        if any(destination.iterdir()) if destination.is_dir() else True:
            raise FileExistsError(f"Refusing to overwrite existing artifact path: {destination}")
    else:
        destination.mkdir(parents=True)
    return destination
