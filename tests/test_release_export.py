"""Tests for deterministic, manifest-driven companion archive export."""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
EXPORTER_PATH = REPO / "tools/build_companion_archives.py"


def load_exporter():
    specification = importlib.util.spec_from_file_location(
        "companion_archive_exporter_test_module", EXPORTER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load companion archive exporter.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


EXPORTER = load_exporter()


class CompanionArchiveExporterTests(unittest.TestCase):
    maxDiff = None

    def test_frozen_public_scope_constants(self) -> None:
        mini = json.loads((REPO / EXPORTER.MINICLIFF_SPEC).read_text(encoding="utf-8"))
        pendulum = json.loads(
            (REPO / EXPORTER.PENDULUM_SPEC).read_text(encoding="utf-8")
        )
        self.assertEqual(
            mini["combined_inventory_sha256"],
            "8d02a0b0ea68f1fe1cae82c35ff1418410f982dc7a0aa36b90d8edecba6df864",
        )
        self.assertEqual(mini["canonical_inputs"]["main"]["file_count"], 116)
        self.assertEqual(
            mini["canonical_inputs"]["ell_sensitivity"]["file_count"], 493
        )
        self.assertEqual(pendulum["companion_data_inventory"]["file_count"], 158)
        self.assertEqual(
            pendulum["companion_data_inventory"]["inventory_sha256"],
            "d1c278aaeaab7224fc3a10f64af0a9c4ab33784adc2dd2c62be5373c616ef01d",
        )

    def test_inventory_canonicalizations_are_distinct(self) -> None:
        mini = {"main/a": {"sha256": "0" * 64, "size": 1}}
        pendulum = {"main/a": {"sha256": "0" * 64, "size_bytes": 1}}
        expected_mini = EXPORTER._compact_json_sha256(
            mini, trailing_newline=True
        )
        expected_pendulum = EXPORTER._compact_json_sha256(
            pendulum, trailing_newline=False
        )
        self.assertEqual(EXPORTER.minicliff_inventory_digest(mini), expected_mini)
        self.assertEqual(
            EXPORTER.pendulum_inventory_digest(pendulum), expected_pendulum
        )
        self.assertNotEqual(expected_mini, expected_pendulum)

    def test_archive_bytes_names_metadata_and_contents_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-export-determinism-") as temporary:
            root = Path(temporary).resolve()
            (root / "nested").mkdir()
            (root / "alpha.txt").write_bytes(b"alpha\n")
            (root / "nested/beta.bin").write_bytes(b"\x00\x01\x02")
            records = {
                name: EXPORTER._record(root, name)
                for name in ("alpha.txt", "nested/beta.bin")
            }
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"
            EXPORTER._write_archive(first, records)
            EXPORTER._write_archive(second, records)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            EXPORTER._validate_archive(first, records)
            with tarfile.open(first, mode="r:gz") as archive:
                members = archive.getmembers()
            self.assertEqual([item.name for item in members], sorted(records))
            for item in members:
                self.assertTrue(item.isfile())
                self.assertEqual(item.mode, 0o644)
                self.assertEqual((item.uid, item.gid), (0, 0))
                self.assertEqual((item.uname, item.gname), ("", ""))
                self.assertEqual(item.mtime, 0)

    def test_archive_validation_rejects_traversal_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-export-unsafe-") as temporary:
            root = Path(temporary)
            traversal = root / "traversal.tar.gz"
            with traversal.open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                    with tarfile.open(
                        fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT
                    ) as archive:
                        info = tarfile.TarInfo("../escape")
                        info.size = 1
                        archive.addfile(info, io.BytesIO(b"x"))
            with self.assertRaisesRegex(EXPORTER.ExportError, "Unsafe archive member"):
                EXPORTER._validate_archive(traversal, {})

            source = root / "a.txt"
            source.write_bytes(b"a")
            record = EXPORTER._record(root.resolve(), "a.txt")
            duplicate = root / "duplicate.tar.gz"
            with duplicate.open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                    with tarfile.open(
                        fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT
                    ) as archive:
                        for _ in range(2):
                            info = tarfile.TarInfo("a.txt")
                            info.size = 1
                            info.mode = 0o644
                            info.mtime = 0
                            archive.addfile(info, io.BytesIO(b"a"))
            with self.assertRaisesRegex(EXPORTER.ExportError, "inventory differs"):
                EXPORTER._validate_archive(duplicate, {"a.txt": record})

    def test_output_guard_rejects_existing_source_overlap_and_symlink_alias(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-export-output-") as temporary:
            parent = Path(temporary).resolve()
            repo = parent / "repository"
            repo.mkdir()
            (repo / ".git").mkdir()
            with self.assertRaisesRegex(EXPORTER.ExportError, "overlap"):
                EXPORTER.resolve_output(repo, repo / "assets")
            existing = parent / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(EXPORTER.ExportError, "must not exist"):
                EXPORTER.resolve_output(repo, existing)
            alias = parent / "alias"
            alias.symlink_to(repo, target_is_directory=True)
            with self.assertRaisesRegex(EXPORTER.ExportError, "overlap"):
                EXPORTER.resolve_output(repo, alias / "assets")

            checkout_alias = parent / "checkout-alias"
            checkout_alias.symlink_to(repo, target_is_directory=True)
            with self.assertRaisesRegex(EXPORTER.ExportError, "non-symlinked"):
                EXPORTER.build_archives(checkout_alias, parent / "new-assets")

    def test_minicliff_selection_excludes_quick_tree(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-export-mini-") as temporary:
            repo = Path(temporary).resolve()
            main = repo / "paper/main"
            sweep = repo / "paper/ell"
            quick = repo / "paper/quick"
            for directory in (main, sweep, quick):
                directory.mkdir(parents=True)
            (main / "main.txt").write_text("main\n", encoding="utf-8")
            (sweep / "ell.txt").write_text("ell\n", encoding="utf-8")
            (quick / "excluded.txt").write_text("quick\n", encoding="utf-8")

            label_rows = {}
            combined = {}
            for label, relative, filename in (
                ("main", "paper/main", "main.txt"),
                ("ell_sensitivity", "paper/ell", "ell.txt"),
            ):
                record = EXPORTER._record(repo, f"{relative}/{filename}")
                row = {"sha256": record.sha256, "size": record.size_bytes}
                inventory = {filename: row}
                label_rows[label] = {
                    "file_count": 1,
                    "total_bytes": record.size_bytes,
                    "inventory_sha256": EXPORTER.minicliff_inventory_digest(inventory),
                }
                combined[f"{label}/{filename}"] = row
            spec = {
                "schema_version": "minicliff.paper_reproducibility.v1",
                "repository_roots": {
                    "main": "paper/main",
                    "ell_sensitivity": "paper/ell",
                },
                "canonical_inputs": label_rows,
                "combined_inventory_sha256": EXPORTER.minicliff_inventory_digest(combined),
            }
            spec_path = repo / EXPORTER.MINICLIFF_SPEC
            spec_path.parent.mkdir(parents=True)
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            records, report = EXPORTER.select_minicliff(repo)
            self.assertEqual(
                sorted(records), ["paper/ell/ell.txt", "paper/main/main.txt"]
            )
            self.assertEqual(report["file_count"], 2)
            self.assertNotIn("paper/quick/excluded.txt", records)

    def test_pendulum_selection_is_manifest_driven_and_excludes_adjacent_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-export-pendulum-") as temporary:
            repo = Path(temporary).resolve()

            def write(relative: str, content: str) -> None:
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            reporting_root = "results/pend/reporting"
            write(f"{reporting_root}/manifest.json", "{}\n")
            write(f"{reporting_root}/raw.csv", "value\n")
            gate = "results/pend/development/aggregated/development_gate.json"
            freeze = "results/pend/development/reporting_freeze_manifest.json"
            write(gate, "{}\n")
            write(freeze, "{}\n")

            supplements = []
            for index in (1, 2):
                root = f"results/pend/supplement_{index}"
                raw_root = f"{root}/raw/seed_0031"
                write(f"{root}/manifest.json", "{}\n")
                write(f"{raw_root}/artifact.bin", f"artifact-{index}\n")
                artifact_hash = EXPORTER.sha256_file(repo / raw_root / "artifact.bin")
                raw_manifest = {"files_sha256": {"artifact.bin": artifact_hash}}
                write(f"{raw_root}/manifest.json", json.dumps(raw_manifest) + "\n")
                write(f"{root}/aggregated/excluded.csv", "excluded\n")
                supplements.append(
                    {
                        "root": root,
                        "raw_manifest": f"{raw_root}/manifest.json",
                    }
                )

            names = [
                f"{reporting_root}/manifest.json",
                f"{reporting_root}/raw.csv",
                gate,
                freeze,
            ]
            for item in supplements:
                names.extend(
                    [
                        f"{item['root']}/manifest.json",
                        item["raw_manifest"],
                        f"{PurePosixPath(item['raw_manifest']).parent.as_posix()}/artifact.bin",
                    ]
                )
            records = {name: EXPORTER._record(repo, name) for name in sorted(names)}
            canonical = {
                name: {"sha256": record.sha256, "size_bytes": record.size_bytes}
                for name, record in records.items()
            }
            inventory = {
                "file_count": len(records),
                "total_bytes": sum(record.size_bytes for record in records.values()),
                "inventory_sha256": EXPORTER.pendulum_inventory_digest(canonical),
                "canonicalization": "fixture canonicalization",
                "scope": "fixture scope",
            }
            spec = {
                "schema": "rvchi2_dqn.pendulum_reproducibility_inputs.v1",
                "canonical_reporting": {
                    "root": reporting_root,
                    "development_gate": gate,
                    "reporting_freeze": freeze,
                },
                "supplemental_studies": supplements,
                "companion_data_inventory": inventory,
            }
            spec_path = repo / EXPORTER.PENDULUM_SPEC
            spec_path.parent.mkdir(parents=True)
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            selected, report = EXPORTER.select_pendulum(repo)
            self.assertEqual(set(selected), set(records))
            self.assertEqual(report["inventory_sha256"], inventory["inventory_sha256"])
            self.assertFalse(any("aggregated/excluded.csv" in name for name in selected))

    def test_selector_rejects_symlinked_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-export-link-") as temporary:
            repo = Path(temporary).resolve()
            real = repo / "real"
            real.mkdir()
            (real / "payload.txt").write_text("payload\n", encoding="utf-8")
            link = repo / "paper/main"
            link.parent.mkdir(parents=True)
            link.symlink_to(real, target_is_directory=True)
            spec_path = repo / EXPORTER.MINICLIFF_SPEC
            spec_path.parent.mkdir(parents=True)
            spec_path.write_text(
                json.dumps(
                    {
                        "schema_version": "minicliff.paper_reproducibility.v1",
                        "repository_roots": {
                            "main": "paper/main",
                            "ell_sensitivity": "paper/main",
                        },
                        "canonical_inputs": {},
                        "combined_inventory_sha256": "",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EXPORTER.ExportError, "Symlink is forbidden"):
                EXPORTER.select_minicliff(repo)


if __name__ == "__main__":
    unittest.main()
