#!/usr/bin/env python3
"""Build and authenticate the two immutable paper companion-data archives.

The archives contain data only.  Member paths are repository-relative so each
archive can be extracted directly at the root of a clean source checkout.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
from typing import Any, Iterable, Mapping, Sequence
import uuid


MINICLIFF_ARCHIVE = "minicliff-companion-data.tar.gz"
PENDULUM_ARCHIVE = "pendulum-companion-data.tar.gz"
MINICLIFF_SPEC = Path("reproducibility/paper_experiments/minicliff/frozen_spec.json")
PENDULUM_SPEC = Path("reproducibility/paper_experiments/pendulum/frozen_inputs.json")
MINICLIFF_WORKFLOW = Path(
    "reproducibility/paper_experiments/minicliff/minicliff_reproduce.py"
)
PENDULUM_WORKFLOW = Path("reproducibility/paper_experiments/pendulum/reproduce.py")
SIDECAR_SCHEMA = "drrl_chi_square.companion_archive.v1"
BUILD_SCHEMA = "drrl_chi_square.companion_archive_build.v1"


class ExportError(RuntimeError):
    """Fail-closed companion-export error."""


@dataclass(frozen=True)
class FileRecord:
    archive_name: str
    source_path: Path
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_json_sha256(value: Any, *, trailing_newline: bool) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if trailing_newline:
        encoded += "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def minicliff_inventory_digest(
    records: Mapping[str, Mapping[str, Any]]
) -> str:
    """Hash MiniCliff's ``sha256``/``size`` inventory with its final newline."""

    return _compact_json_sha256(records, trailing_newline=True)


def pendulum_inventory_digest(
    records: Mapping[str, Mapping[str, Any]]
) -> str:
    """Hash Pendulum's ``sha256``/``size_bytes`` inventory without a newline."""

    return _compact_json_sha256(records, trailing_newline=False)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"Cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExportError(f"Expected a JSON object in {path}.")
    return value


def _validate_archive_name(name: str) -> str:
    if not name or "\x00" in name or "\\" in name:
        raise ExportError(f"Unsafe archive member name: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ExportError(f"Unsafe archive member name: {name!r}")
    normalized = pure.as_posix()
    if normalized != name:
        raise ExportError(f"Non-canonical archive member name: {name!r}")
    return normalized


def _resolve_repo_file(repo_root: Path, relative: str) -> Path:
    relative = _validate_archive_name(relative)
    candidate = repo_root
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ExportError(f"Symlink is forbidden in companion data: {candidate}")
    try:
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise ExportError(f"Missing companion-data file {candidate}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise ExportError(f"Companion-data entry is not a regular file: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ExportError(f"Companion-data path escapes the repository: {relative}") from exc
    return resolved


def _record(repo_root: Path, relative: str) -> FileRecord:
    path = _resolve_repo_file(repo_root, relative)
    return FileRecord(
        archive_name=relative,
        source_path=path,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _regular_files_under(repo_root: Path, relative_root: str) -> list[str]:
    _validate_archive_name(relative_root)
    root = repo_root
    for part in PurePosixPath(relative_root).parts:
        root = root / part
        if root.is_symlink():
            raise ExportError(f"Symlink is forbidden in companion data: {root}")
    try:
        root.resolve().relative_to(repo_root)
    except ValueError as exc:
        raise ExportError(f"Companion-data root escapes the repository: {relative_root}") from exc
    if root.is_symlink() or not root.is_dir():
        raise ExportError(f"Expected a real companion-data directory: {root}")
    result: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ExportError(f"Symlink is forbidden in companion data: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ExportError(f"Special node is forbidden in companion data: {path}")
        result.append(path.relative_to(repo_root).as_posix())
    return result


def _insert(records: dict[str, FileRecord], record: FileRecord) -> None:
    if record.archive_name in records:
        raise ExportError(f"Duplicate archive path: {record.archive_name}")
    records[record.archive_name] = record


def select_minicliff(repo_root: Path) -> tuple[dict[str, FileRecord], dict[str, Any]]:
    spec_path = _resolve_repo_file(repo_root, MINICLIFF_SPEC.as_posix())
    spec = _load_json(spec_path)
    if spec.get("schema_version") != "minicliff.paper_reproducibility.v1":
        raise ExportError("Unsupported MiniCliff frozen specification.")

    records: dict[str, FileRecord] = {}
    combined: dict[str, dict[str, Any]] = {}
    study_report: dict[str, Any] = {}
    for label in ("main", "ell_sensitivity"):
        relative_root = spec["repository_roots"].get(label)
        if not isinstance(relative_root, str):
            raise ExportError(f"MiniCliff specification lacks the {label} root.")
        label_inventory: dict[str, dict[str, Any]] = {}
        for relative in _regular_files_under(repo_root, relative_root):
            record = _record(repo_root, relative)
            _insert(records, record)
            inside = PurePosixPath(relative).relative_to(PurePosixPath(relative_root))
            row = {"sha256": record.sha256, "size": record.size_bytes}
            label_inventory[inside.as_posix()] = row
            combined[f"{label}/{inside.as_posix()}"] = row
        expected = spec["canonical_inputs"][label]
        observed_count = len(label_inventory)
        observed_bytes = sum(row["size"] for row in label_inventory.values())
        observed_digest = minicliff_inventory_digest(label_inventory)
        if (
            observed_count != expected["file_count"]
            or observed_bytes != expected["total_bytes"]
            or observed_digest != expected["inventory_sha256"]
        ):
            raise ExportError(
                f"MiniCliff {label} inventory differs: "
                f"count={observed_count}, bytes={observed_bytes}, sha256={observed_digest}"
            )
        study_report[label] = {
            "file_count": observed_count,
            "total_bytes": observed_bytes,
            "inventory_sha256": observed_digest,
        }

    combined_digest = minicliff_inventory_digest(combined)
    if combined_digest != spec["combined_inventory_sha256"]:
        raise ExportError(f"MiniCliff combined inventory differs: {combined_digest}")
    total_bytes = sum(record.size_bytes for record in records.values())
    return records, {
        "file_count": len(records),
        "total_bytes": total_bytes,
        "inventory_sha256": combined_digest,
        "inventory_canonicalization": (
            "compact sorted JSON mapping main/<path> and ell_sensitivity/<path> "
            "to {sha256,size}, followed by one newline"
        ),
        "studies": study_report,
    }


def select_pendulum(repo_root: Path) -> tuple[dict[str, FileRecord], dict[str, Any]]:
    spec_path = _resolve_repo_file(repo_root, PENDULUM_SPEC.as_posix())
    spec = _load_json(spec_path)
    if spec.get("schema") != "rvchi2_dqn.pendulum_reproducibility_inputs.v1":
        raise ExportError("Unsupported Pendulum frozen specification.")
    canonical = spec.get("canonical_reporting")
    supplements = spec.get("supplemental_studies")
    if not isinstance(canonical, dict) or not isinstance(supplements, list):
        raise ExportError("Pendulum companion-data declaration is malformed.")

    selected_names: list[str] = []
    reporting_root = canonical.get("root")
    if not isinstance(reporting_root, str):
        raise ExportError("Pendulum canonical reporting root is absent.")
    selected_names.extend(_regular_files_under(repo_root, reporting_root))
    for key in ("development_gate", "reporting_freeze"):
        relative = canonical.get(key)
        if not isinstance(relative, str):
            raise ExportError(f"Pendulum canonical declaration lacks {key}.")
        selected_names.append(relative)

    for item in supplements:
        if not isinstance(item, dict):
            raise ExportError("Pendulum supplemental-study declaration is malformed.")
        study_root = item.get("root")
        raw_manifest_relative = item.get("raw_manifest")
        if not isinstance(study_root, str) or not isinstance(raw_manifest_relative, str):
            raise ExportError("Pendulum supplemental-study paths are malformed.")
        selected_names.append(f"{study_root}/manifest.json")
        selected_names.append(raw_manifest_relative)
        raw_manifest = _load_json(_resolve_repo_file(repo_root, raw_manifest_relative))
        declared = raw_manifest.get("files_sha256")
        if not isinstance(declared, dict) or not declared:
            raise ExportError(f"Raw manifest has no file declaration: {raw_manifest_relative}")
        raw_root = PurePosixPath(raw_manifest_relative).parent
        expected_raw_names = {PurePosixPath(raw_manifest_relative).name}
        for name in declared:
            if not isinstance(name, str) or PurePosixPath(name).name != name:
                raise ExportError(f"Unsafe raw-manifest filename: {name!r}")
            expected_raw_names.add(name)
            selected_names.append((raw_root / name).as_posix())
        raw_entries = list((repo_root / raw_root.as_posix()).iterdir())
        unsafe_raw_entries = [
            path.name
            for path in raw_entries
            if path.is_symlink() or not path.is_file()
        ]
        if unsafe_raw_entries:
            raise ExportError(
                f"Raw bundle contains symlinks or special entries at {raw_root}: "
                f"{sorted(unsafe_raw_entries)}"
            )
        actual_raw_names = {path.name for path in raw_entries}
        if actual_raw_names != expected_raw_names:
            raise ExportError(
                f"Raw bundle differs at {raw_root}: expected={sorted(expected_raw_names)}, "
                f"observed={sorted(actual_raw_names)}"
            )

    records: dict[str, FileRecord] = {}
    for relative in sorted(selected_names):
        _insert(records, _record(repo_root, relative))
    canonical_records = {
        name: {"sha256": record.sha256, "size_bytes": record.size_bytes}
        for name, record in sorted(records.items())
    }
    observed = {
        "file_count": len(records),
        "total_bytes": sum(record.size_bytes for record in records.values()),
        "inventory_sha256": pendulum_inventory_digest(canonical_records),
    }
    expected = spec["companion_data_inventory"]
    for key, value in observed.items():
        if value != expected[key]:
            raise ExportError(
                f"Pendulum companion {key} differs: expected={expected[key]}, observed={value}"
            )
    return records, {
        **observed,
        "inventory_canonicalization": expected["canonicalization"],
        "scope": expected["scope"],
    }


def _write_archive(path: Path, records: Mapping[str, FileRecord]) -> None:
    with path.open("xb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
            ) as archive:
                for name in sorted(records):
                    record = records[name]
                    info = tarfile.TarInfo(_validate_archive_name(name))
                    info.size = record.size_bytes
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.type = tarfile.REGTYPE
                    with record.source_path.open("rb") as source:
                        archive.addfile(info, source)


def _validate_archive(path: Path, expected: Mapping[str, FileRecord]) -> None:
    expected_names = sorted(expected)
    observed_names: list[str] = []
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            name = _validate_archive_name(member.name)
            observed_names.append(name)
            if not member.isfile():
                raise ExportError(f"Archive member is not a regular file: {name}")
            if (
                member.mode != 0o644
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mtime != 0
            ):
                raise ExportError(f"Archive metadata is not normalized: {name}")
            record = expected.get(name)
            if record is None:
                raise ExportError(f"Unexpected archive member: {name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ExportError(f"Cannot read archive member: {name}")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            if size != record.size_bytes or digest.hexdigest() != record.sha256:
                raise ExportError(f"Archive member content differs: {name}")
    if observed_names != expected_names or len(observed_names) != len(set(observed_names)):
        raise ExportError(
            "Archive member inventory differs or is not sorted: "
            f"expected={expected_names}, observed={observed_names}"
        )


def _sidecar(
    *, archive_path: Path, payload: Mapping[str, Any], spec_relative: Path, repo_root: Path
) -> dict[str, Any]:
    return {
        "schema_version": SIDECAR_SCHEMA,
        "archive_filename": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "payload": dict(payload),
        "source_spec": {
            "path": spec_relative.as_posix(),
            "sha256": sha256_file(repo_root / spec_relative),
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_authentication(repo_root: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    commands = (
        [sys.executable, str(repo_root / MINICLIFF_WORKFLOW), "verify"],
        [
            sys.executable,
            str(repo_root / PENDULUM_WORKFLOW),
            "verify",
            "--require-live-correspondence",
        ],
    )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ExportError(
                f"Source authentication failed for {Path(command[1]).name}: {detail}"
            )


def _path_overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def resolve_output(repo_root: Path, requested: Path) -> Path:
    destination = requested.expanduser().resolve(strict=False)
    if requested.exists() or requested.is_symlink() or destination.exists():
        raise ExportError(f"Output directory must not exist: {requested}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ExportError(f"Output parent must be a real existing directory: {destination.parent}")
    if _path_overlaps(destination, repo_root):
        raise ExportError("Output directory may not overlap the source repository.")
    return destination


def _fsync_tree(root: Path) -> None:
    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_archives(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    requested_repo_root = repo_root.expanduser()
    if requested_repo_root.is_symlink():
        raise ExportError(f"Expected a real, non-symlinked Git checkout: {repo_root}")
    repo_root = requested_repo_root.resolve()
    if not (repo_root / ".git").is_dir():
        raise ExportError(f"Expected a real Git checkout: {repo_root}")
    destination = resolve_output(repo_root, output_dir)
    _run_authentication(repo_root)
    mini_before, mini_payload = select_minicliff(repo_root)
    pend_before, pend_payload = select_pendulum(repo_root)

    stage = destination.with_name(f".{destination.name}.stage-{uuid.uuid4().hex}")
    if stage.exists() or stage.is_symlink():
        raise ExportError(f"Unexpected staging collision: {stage}")
    stage.mkdir(mode=0o755)
    try:
        archive_inputs = (
            (MINICLIFF_ARCHIVE, mini_before, mini_payload, MINICLIFF_SPEC),
            (PENDULUM_ARCHIVE, pend_before, pend_payload, PENDULUM_SPEC),
        )
        sidecars: dict[str, dict[str, Any]] = {}
        for filename, records, payload, spec_relative in archive_inputs:
            archive_path = stage / filename
            _write_archive(archive_path, records)
            _validate_archive(archive_path, records)
            sidecar = _sidecar(
                archive_path=archive_path,
                payload=payload,
                spec_relative=spec_relative,
                repo_root=repo_root,
            )
            _write_json(stage / f"{filename}.json", sidecar)
            sidecars[filename] = sidecar

        checksum_lines = [
            f"{sidecars[name]['archive_sha256']}  {name}" for name in sorted(sidecars)
        ]
        (stage / "SHA256SUMS").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )

        mini_after, _ = select_minicliff(repo_root)
        pend_after, _ = select_pendulum(repo_root)
        if mini_after != mini_before or pend_after != pend_before:
            raise ExportError("Authenticated source content changed during archive construction.")
        _run_authentication(repo_root)
        _fsync_tree(stage)
        if destination.exists() or destination.is_symlink():
            raise ExportError(f"Output appeared before publication: {destination}")
        os.replace(stage, destination)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    return {
        "schema_version": BUILD_SCHEMA,
        "output_directory": str(destination),
        "archives": sidecars,
        "sha256sums": "SHA256SUMS",
        "source_reauthenticated_after_build": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="source checkout containing the authenticated companion data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="fresh, nonexistent directory outside the source checkout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = build_archives(arguments.repo_root, arguments.output_dir)
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
