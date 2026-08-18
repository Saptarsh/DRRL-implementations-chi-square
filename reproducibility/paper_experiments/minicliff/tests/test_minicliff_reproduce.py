"""Focused tests for the isolated MiniCliff reproducibility capsule."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


CAPSULE = Path(__file__).resolve().parents[1]
REPO = CAPSULE.parents[2]
WORKFLOW = CAPSULE / "minicliff_reproduce.py"


def load_workflow():
    specification = importlib.util.spec_from_file_location(
        "isolated_minicliff_reproduce_test_module", WORKFLOW
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load MiniCliff workflow.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


WORKFLOW_MODULE = load_workflow()


class MiniCliffReproducibilityTests(unittest.TestCase):
    maxDiff = None

    def run_cli(self, *arguments: str) -> tuple[subprocess.CompletedProcess[str], dict]:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, str(WORKFLOW), *arguments],
            cwd=REPO,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        payload = json.loads(completed.stdout) if completed.stdout.strip().startswith("{") else {}
        return completed, payload

    def test_frozen_snapshot_is_exact_and_live_sources_correspond(self) -> None:
        spec = WORKFLOW_MODULE.load_spec()
        WORKFLOW_MODULE.verify_snapshot(spec, require_live_match=True)
        expected = spec["source_snapshot"]
        self.assertEqual(
            set(expected),
            {
                path.relative_to(WORKFLOW_MODULE.SNAPSHOT_ROOT).as_posix()
                for path in WORKFLOW_MODULE.SNAPSHOT_ROOT.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            },
        )

    def test_runtime_mismatch_helper_is_exact_and_structured(self) -> None:
        spec = WORKFLOW_MODULE.load_spec()
        actual = dict(WORKFLOW_MODULE.frozen_runtime_requirements(spec))
        self.assertEqual(
            WORKFLOW_MODULE.frozen_runtime_mismatches(spec, actual), {}
        )
        actual["platform"] = "compatible-test-platform"
        self.assertEqual(
            WORKFLOW_MODULE.frozen_runtime_mismatches(spec, actual),
            {
                "platform": {
                    "expected": spec["audit_environment"]["platform"],
                    "actual": "compatible-test-platform",
                }
            },
        )

    def test_runtime_drift_is_fail_closed_without_portable_opt_in(self) -> None:
        spec = WORKFLOW_MODULE.load_spec()
        drifted = dict(WORKFLOW_MODULE.frozen_runtime_requirements(spec))
        drifted["platform"] = "compatible-test-platform"
        with tempfile.TemporaryDirectory(prefix="minicliff-runtime-guard-") as temporary:
            output = Path(temporary) / "candidate"
            with mock.patch.object(
                WORKFLOW_MODULE,
                "authenticate",
                return_value={"combined_inventory_sha256": "test"},
            ), mock.patch.object(
                WORKFLOW_MODULE, "runtime_environment", return_value=drifted
            ):
                with self.assertRaisesRegex(
                    WORKFLOW_MODULE.ReproducibilityError,
                    "Runtime differs from the frozen audit environment",
                ):
                    WORKFLOW_MODULE.regenerate(output)
            self.assertFalse(output.exists())

    def test_cli_help_scopes_runtime_drift_to_regeneration(self) -> None:
        regeneration_help, _ = self.run_cli("regenerate", "--help")
        training_help, _ = self.run_cli("train", "--help")
        self.assertEqual(regeneration_help.returncode, 0, regeneration_help.stderr)
        self.assertEqual(training_help.returncode, 0, training_help.stderr)
        self.assertIn("--allow-runtime-drift", regeneration_help.stdout)
        self.assertNotIn("--allow-runtime-drift", training_help.stdout)

    def test_canonical_verify_is_read_only_and_reconstructs_every_run(self) -> None:
        spec = WORKFLOW_MODULE.load_spec()
        before = {
            label: WORKFLOW_MODULE.canonical_hash(
                WORKFLOW_MODULE.inventory(WORKFLOW_MODULE.study_root(spec, label))
            )
            for label in ("main", "ell_sensitivity")
        }
        completed, report = self.run_cli("verify")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(report["authenticated"])
        self.assertEqual(report["canonical_inputs"]["main"]["runs_validated"], 25)
        self.assertEqual(
            report["canonical_inputs"]["ell_sensitivity"]["runs_validated"], 120
        )
        self.assertTrue(report["cross_study_ell_0p1_seeds_1_through_20_exact"])
        after = {
            label: WORKFLOW_MODULE.canonical_hash(
                WORKFLOW_MODULE.inventory(WORKFLOW_MODULE.study_root(spec, label))
            )
            for label in ("main", "ell_sensitivity")
        }
        self.assertEqual(before, after)

    def test_inventory_rejects_coordinated_tree_copy_tamper(self) -> None:
        spec = WORKFLOW_MODULE.load_spec()
        source = WORKFLOW_MODULE.study_root(spec, "main")
        with tempfile.TemporaryDirectory(prefix="minicliff-inventory-test-") as temporary:
            copied = Path(temporary) / "main"
            shutil.copytree(source, copied)
            WORKFLOW_MODULE.verify_inventory(copied, spec["canonical_inputs"]["main"])
            target = copied / "aggregated/ell_summary.csv"
            target.write_bytes(target.read_bytes() + b"\n")
            with self.assertRaises(WORKFLOW_MODULE.ReproducibilityError):
                WORKFLOW_MODULE.verify_inventory(
                    copied, spec["canonical_inputs"]["main"]
                )

    def test_output_guard_rejects_historical_overlap_and_symlink_alias(self) -> None:
        historical = WORKFLOW_MODULE.HISTORICAL_ROOT.resolve()
        for candidate in (
            historical,
            historical / "main/rebuild",
            historical.parent,
            WORKFLOW_MODULE.CAPSULE_ROOT / "candidate",
        ):
            with self.assertRaises(WORKFLOW_MODULE.ReproducibilityError):
                WORKFLOW_MODULE.resolve_new_destination(candidate)
        with tempfile.TemporaryDirectory(prefix="minicliff-alias-test-") as temporary:
            alias = Path(temporary) / "history_alias"
            alias.symlink_to(historical, target_is_directory=True)
            with self.assertRaises(WORKFLOW_MODULE.ReproducibilityError):
                WORKFLOW_MODULE.resolve_new_destination(alias / "candidate")

    def test_training_requires_literal_expensive_acknowledgement_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="minicliff-training-guard-") as temporary:
            output = Path(temporary) / "training"
            with self.assertRaises(WORKFLOW_MODULE.ReproducibilityError):
                WORKFLOW_MODULE.train(output, "main", "no")
            self.assertFalse(output.exists())

    def test_fresh_copy_regeneration_and_immutable_reauthentication(self) -> None:
        spec = WORKFLOW_MODULE.load_spec()
        canonical_before = spec["combined_inventory_sha256"]
        with tempfile.TemporaryDirectory(prefix="minicliff-regeneration-test-") as temporary:
            output = Path(temporary) / "candidate"
            completed, report = self.run_cli(
                "regenerate", "--output-root", str(output)
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(report["training_performed"])
            self.assertFalse(report["reporting_or_selection_authority"])
            self.assertEqual(
                report["schema_version"], WORKFLOW_MODULE.STRICT_REGENERATION_SCHEMA
            )
            self.assertTrue(report["runtime_matches_frozen"])
            self.assertEqual(report["frozen_runtime_mismatches"], {})
            self.assertTrue(
                report["paper_composite_png_pixel_identical_to_canonical"]
            )
            self.assertTrue(
                all(
                    all(checks.values())
                    for checks in report[
                        "aggregate_matches_canonical_bytes_or_values"
                    ].values()
                )
            )
            tree_before = {
                path.relative_to(output).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in output.rglob("*")
                if path.is_file()
            }
            verified, verification = self.run_cli(
                "verify-output", "--output-root", str(output)
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertTrue(verification["verified"])
            tree_after = {
                path.relative_to(output).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(tree_before, tree_after)
            self.assertEqual(
                verification["canonical_combined_inventory_sha256"], canonical_before
            )
            manifest_path = output / WORKFLOW_MODULE.REPRODUCTION_MANIFEST
            manifest = WORKFLOW_MODULE.load_json(manifest_path)
            manifest["paper_composite_png_pixel_identical_to_canonical"] = False
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                WORKFLOW_MODULE, "png_pixels_equal", return_value=False
            ):
                with self.assertRaisesRegex(
                    WORKFLOW_MODULE.ReproducibilityError,
                    "PNG pixels differ from canonical",
                ):
                    WORKFLOW_MODULE.verify_output(output)

    def test_portable_output_records_drift_without_weakening_aggregate_checks(self) -> None:
        spec = WORKFLOW_MODULE.load_spec()
        runtime = dict(WORKFLOW_MODULE.frozen_runtime_requirements(spec))
        runtime["platform"] = "compatible-test-platform"
        mismatches = WORKFLOW_MODULE.frozen_runtime_mismatches(spec, runtime)
        canonical_png = (
            WORKFLOW_MODULE.study_root(spec, "main")
            / "figures/tabular_tac_composite.png"
        ).resolve()
        actual_pixels_equal = WORKFLOW_MODULE.png_pixels_equal

        def simulate_runtime_render_drift(first: Path, second: Path) -> bool:
            if Path(second).resolve() == canonical_png:
                return False
            return actual_pixels_equal(Path(first), Path(second))

        with tempfile.TemporaryDirectory(prefix="minicliff-portable-test-") as temporary:
            output = Path(temporary) / "candidate"
            with mock.patch.object(
                WORKFLOW_MODULE, "runtime_environment", return_value=runtime
            ), mock.patch.object(
                WORKFLOW_MODULE,
                "png_pixels_equal",
                side_effect=simulate_runtime_render_drift,
            ):
                manifest = WORKFLOW_MODULE.regenerate(
                    output, allow_runtime_drift=True
                )
                self.assertEqual(
                    manifest["schema_version"],
                    WORKFLOW_MODULE.PORTABLE_REGENERATION_SCHEMA,
                )
                self.assertFalse(manifest["runtime_matches_frozen"])
                self.assertEqual(manifest["frozen_runtime_mismatches"], mismatches)
                self.assertFalse(
                    manifest["paper_composite_png_pixel_identical_to_canonical"]
                )
                verification = WORKFLOW_MODULE.verify_output(output)
            self.assertTrue(verification["verified"])
            self.assertFalse(verification["runtime_matches_frozen"])
            self.assertEqual(verification["frozen_runtime_mismatches"], mismatches)
            self.assertFalse(
                verification["paper_composite_png_pixel_identical_to_canonical"]
            )
            self.assertTrue(
                verification["paper_composite_png_matches_fresh_render"]
            )

            manifest_path = output / WORKFLOW_MODULE.REPRODUCTION_MANIFEST
            manifest = WORKFLOW_MODULE.load_json(manifest_path)
            manifest["frozen_runtime_mismatches"] = {}
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                WORKFLOW_MODULE.ReproducibilityError,
                "runtime-mismatch record is internally inconsistent",
            ):
                WORKFLOW_MODULE.verify_output(output)

            manifest["frozen_runtime_mismatches"] = mismatches
            png_path = output / "paper_derivative/tabular_tac_composite.png"
            with WORKFLOW_MODULE.Image.open(png_path) as image:
                tampered = image.convert("RGBA")
            original_pixel = tampered.getpixel((0, 0))
            tampered.putpixel(
                (0, 0),
                tuple((value + 1) % 256 for value in original_pixel),
            )
            tampered.save(png_path)
            manifest["payload_inventory"] = WORKFLOW_MODULE.payload_inventory(output)
            manifest["payload_inventory_sha256"] = WORKFLOW_MODULE.canonical_hash(
                manifest["payload_inventory"]
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                WORKFLOW_MODULE, "runtime_environment", return_value=runtime
            ), mock.patch.object(
                WORKFLOW_MODULE,
                "png_pixels_equal",
                side_effect=simulate_runtime_render_drift,
            ):
                with self.assertRaisesRegex(
                    WORKFLOW_MODULE.ReproducibilityError,
                    "differs from a fresh render",
                ):
                    WORKFLOW_MODULE.verify_output(output)

    def test_output_verification_rejects_unbound_extra_file(self) -> None:
        # A minimal fabricated output reaches the inventory check and must fail;
        # it cannot acquire authority merely by naming a recognized schema.
        with tempfile.TemporaryDirectory(prefix="minicliff-output-tamper-") as temporary:
            root = Path(temporary) / "candidate"
            root.mkdir()
            (root / "payload.txt").write_text("payload\n", encoding="utf-8")
            (root / "reproduction_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "minicliff.regenerated_from_authenticated_raw.v1",
                        "reporting_or_selection_authority": False,
                        "payload_inventory": {},
                        "payload_inventory_sha256": WORKFLOW_MODULE.canonical_hash({}),
                    }
                ),
                encoding="utf-8",
            )
            completed, _ = self.run_cli(
                "verify-output", "--output-root", str(root)
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("payload inventory differs", completed.stderr)

    def test_tag_audit_declares_companion_data_requirement(self) -> None:
        report = WORKFLOW_MODULE.tag_audit()
        self.assertTrue(report["companion_data_required"])
        self.assertEqual(
            report["companion_data_combined_inventory_sha256"],
            WORKFLOW_MODULE.load_spec()["combined_inventory_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
