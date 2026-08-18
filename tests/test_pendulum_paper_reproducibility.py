from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import os
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "reproducibility" / "paper_experiments" / "pendulum" / "reproduce.py"
)
SPEC = importlib.util.spec_from_file_location("pendulum_paper_reproduce", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


class PendulumPaperReproducibilityTests(unittest.TestCase):
    def test_frozen_snapshot_is_exact_and_live_correspondence_is_optional(self) -> None:
        spec = reproduce.load_input_spec()
        strict = reproduce.verify_snapshot(spec, require_live_correspondence=True)
        self.assertEqual(strict["snapshot_file_count"], 16)
        self.assertTrue(strict["all_live_files_match"])
        self.assertEqual(
            {row["relative_path"] for row in strict["snapshot_files"]},
            set(spec["source_snapshot"]),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            live = root / "live"
            snapshot.mkdir()
            live.mkdir()
            (snapshot / "code.py").write_text("frozen\n", encoding="utf-8")
            (live / "code.py").write_text("drifted\n", encoding="utf-8")
            miniature = {
                "source_snapshot": {"code.py": reproduce._sha256(snapshot / "code.py")},
                "live_source_correspondence": {"code.py": "live/code.py"},
            }
            report = reproduce.verify_snapshot(
                miniature, repo_root=root, snapshot_root=snapshot
            )
            self.assertFalse(report["all_live_files_match"])
            with self.assertRaisesRegex(reproduce.VerificationError, "Live-source drift"):
                reproduce.verify_snapshot(
                    miniature,
                    repo_root=root,
                    snapshot_root=snapshot,
                    require_live_correspondence=True,
                )
            rogue = snapshot / "__pycache__" / "code.pyc"
            rogue.parent.mkdir()
            rogue.write_bytes(b"not-authenticated")
            with self.assertRaisesRegex(reproduce.VerificationError, "inventory differs"):
                reproduce.verify_snapshot(
                    miniature, repo_root=root, snapshot_root=snapshot
                )
            rogue.unlink()
            rogue.parent.rmdir()
            link = snapshot / "linked.py"
            link.symlink_to(snapshot / "code.py")
            with self.assertRaisesRegex(reproduce.VerificationError, "symlinks"):
                reproduce.verify_snapshot(
                    miniature, repo_root=root, snapshot_root=snapshot
                )

    def test_frozen_canonical_inventory_and_appendices_verify(self) -> None:
        report = reproduce.verify_all_inputs()
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["inventory"]["declared_file_count"], 145)
        self.assertEqual(report["inventory"]["canonical_tree_file_count"], 132)
        self.assertEqual(len(report["raw_reporting_bundles"]), 10)
        self.assertEqual(report["frozen_snapshot"]["snapshot_file_count"], 16)
        self.assertTrue(report["frozen_snapshot"]["all_live_files_match"])
        self.assertEqual(report["companion_data_inventory"]["file_count"], 158)
        self.assertEqual(
            report["companion_data_inventory"]["inventory_sha256"],
            "d1c278aaeaab7224fc3a10f64af0a9c4ab33784adc2dd2c62be5373c616ef01d",
        )
        self.assertTrue(
            all(row["artifact_count"] == 10 for row in report["raw_reporting_bundles"])
        )
        self.assertEqual(
            {row["label"] for row in report["supplemental_studies"]},
            {"exact_inner_seed31_appendix", "full_nn_seed31_appendix"},
        )

    def test_output_guard_rejects_canonical_related_and_existing_paths(self) -> None:
        canonical = (
            ROOT / "results" / "rvchi2_dqn" / "v2" / "pendulum" / "reporting_full_nn_v1"
        )
        with self.assertRaises(reproduce.VerificationError):
            reproduce.assert_fresh_output(canonical, canonical)
        with self.assertRaises(reproduce.VerificationError):
            reproduce.assert_fresh_output(canonical / "new", canonical)
        with self.assertRaises(reproduce.VerificationError):
            reproduce.assert_fresh_output(canonical.parent, canonical)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(reproduce.VerificationError):
                reproduce.assert_fresh_output(Path(directory), canonical)
            fresh = Path(directory) / "fresh"
            self.assertEqual(reproduce.assert_fresh_output(fresh, canonical), fresh.resolve())

    def test_inventory_verifier_rejects_tamper_and_extra_canonical_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical"
            canonical.mkdir()
            payload = canonical / "value.txt"
            payload.write_text("frozen\n", encoding="utf-8")
            inventory = canonical / "inventory.json"
            entry = {
                "path": "canonical/value.txt",
                "sha256": reproduce._sha256(payload),
                "size_bytes": payload.stat().st_size,
            }
            inventory.write_text(
                json.dumps(
                    {
                        "file_inventory": {"file_count": 1, "files": [entry]},
                        "authorization": {"status": "authenticated"},
                        "deep_validation": {"status": "passed"},
                    }
                ),
                encoding="utf-8",
            )
            digest = reproduce._sha256(inventory)
            verified = reproduce.verify_inventory(root, inventory, digest, canonical)
            self.assertEqual(verified["declared_file_count"], 1)
            payload.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(reproduce.VerificationError):
                reproduce.verify_inventory(root, inventory, digest, canonical)
            payload.write_text("frozen\n", encoding="utf-8")
            (canonical / "rogue.txt").write_text("rogue", encoding="utf-8")
            with self.assertRaises(reproduce.VerificationError):
                reproduce.verify_inventory(root, inventory, digest, canonical)

    def test_full_rerun_is_fail_closed_without_literal_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "expensive"
            with self.assertRaisesRegex(reproduce.VerificationError, "disabled"):
                reproduce.full_tuned_rerun(
                    output,
                    confirmation="yes",
                    max_parallel=1,
                )
            self.assertFalse(output.exists())

    def test_cli_help_labels_full_rerun_as_optional_and_expensive(self) -> None:
        parser = reproduce._parser()
        help_text = parser.format_help()
        self.assertIn("OPTIONAL/EXPENSIVE", help_text)
        self.assertIn("full-rerun", help_text)
        self.assertIn("tag-audit", help_text)

    def test_tag_audit_reports_external_companion_boundary(self) -> None:
        report = reproduce.tag_audit()
        self.assertTrue(report["companion_data_required"])
        self.assertEqual(report["frozen_snapshot_file_count"], 16)
        self.assertEqual(
            report["companion_data"][0]["inventory_sha256"],
            "cbd473f2a0fcbfb165cd3e6ecbd1852638c03adb1f1687d9243ef2f99a8b133c",
        )
        self.assertEqual(report["companion_data_inventory"]["file_count"], 158)
        self.assertEqual(report["companion_data_inventory"]["total_bytes"], 36_446_249)
        self.assertEqual(
            report["tag_ready"],
            not report["untracked_capsule_files"]
            and not report["dirty_tracked_capsule_files"],
        )

    def test_snapshot_runner_help_does_not_dirty_snapshot(self) -> None:
        spec = reproduce.load_input_spec()
        before = reproduce.verify_snapshot(spec)
        runner = reproduce.SNAPSHOT_ROOT / spec["executables"]["runner"]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = reproduce._run(
            [sys.executable, str(runner), "--help"],
            cwd=reproduce.SNAPSHOT_ROOT,
            environment=environment,
        )
        self.assertEqual(completed["returncode"], 0)
        after = reproduce.verify_snapshot(spec)
        self.assertEqual(
            before["snapshot_inventory_sha256"], after["snapshot_inventory_sha256"]
        )

    def test_child_command_output_is_captured_not_forwarded(self) -> None:
        record = reproduce._run(
            [sys.executable, "-c", "print('captured-child-output')"],
            cwd=ROOT,
            environment=os.environ,
        )
        self.assertEqual(record["returncode"], 0)
        self.assertEqual(record["stdout"], "captured-child-output\n")


if __name__ == "__main__":
    unittest.main()
