from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_SHA256.csv"
MANIFEST_SIDECAR = ROOT / "MANIFEST_SHA256.csv.sha256"
EXPECTED_CHECKPOINT_HASH = "4b17c076a4121dfe3233a28214a81b4d0a1ee1b53209e2f668b960a6bc62b00a"
EXPECTED_STARTS = [162666, 808000, 1453333]
EXPECTED_DEFAULT_THRESHOLD = 0.2287546408673127
EXPECTED_CONSERVATIVE_THRESHOLD = 0.6510023052576646


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_app():
    app_path = ROOT / "app.py"
    spec = importlib.util.spec_from_file_location("conextbimamba_deployed_app", app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {app_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_manifest() -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    failures: list[str] = []
    checks["manifest_exists"] = MANIFEST.is_file()
    checks["manifest_sidecar_exists"] = MANIFEST_SIDECAR.is_file()
    if not all(checks.values()):
        return checks, ["The controlled package manifest or its sidecar is missing."]

    sidecar_tokens = MANIFEST_SIDECAR.read_text(encoding="utf-8").strip().split()
    checks["manifest_sidecar_matches"] = bool(sidecar_tokens) and sidecar_tokens[0].lower() == sha256(MANIFEST)
    if not checks["manifest_sidecar_matches"]:
        failures.append("MANIFEST_SHA256.csv does not match its SHA-256 sidecar.")

    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checks["manifest_has_rows"] = bool(rows)
    for row in rows:
        relative = row["relative_path"]
        path = ROOT / Path(relative)
        exists = path.is_file()
        size_ok = exists and path.stat().st_size == int(row["size_bytes"])
        hash_ok = exists and sha256(path) == row["sha256"].lower()
        if not (exists and size_ok and hash_ok):
            failures.append(
                f"Controlled file mismatch: {relative} (exists={exists}, size_ok={size_ok}, hash_ok={hash_ok})"
            )
    checks["all_controlled_files_match"] = not failures
    checks["controlled_file_count_positive"] = len(rows) >= 20
    return checks, failures


def verify_contract(app: Any, smoke: bool) -> tuple[dict[str, bool], dict[str, Any]]:
    checkpoint = ROOT / app.FROZEN_CHECKPOINT
    table = ROOT / "assets/corrected_benchmark.csv"
    long_audio = ROOT / "examples/wavs/demo_ship_300s.wav"
    short_audio = ROOT / "examples/wavs/smoke_test_5s.wav"
    checks = {
        "app_version": app.APP_VERSION == "1.0.0-windows-deploy",
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_hash": checkpoint.is_file() and sha256(checkpoint) == EXPECTED_CHECKPOINT_HASH,
        "benchmark_sample_rate": int(app.BENCHMARK_SAMPLE_RATE) == 16000,
        "benchmark_bag_samples": int(app.BENCHMARK_BAG_SAMPLES) == 1_936_000,
        "frozen_segment_samples": int(app.FROZEN_SEGMENT_SAMPLES) == 320_000,
        "frozen_starts": list(app.FROZEN_SEGMENT_STARTS) == EXPECTED_STARTS,
        "default_threshold": abs(app.MODES["ConExtBiMamba-Default"].threshold - EXPECTED_DEFAULT_THRESHOLD) < 1e-15,
        "default_mean_aggregation": app.MODES["ConExtBiMamba-Default"].aggregation == "mean",
        "conservative_threshold": abs(
            app.MODES["ConExtBiMamba-Conservative"].threshold - EXPECTED_CONSERVATIVE_THRESHOLD
        )
        < 1e-15,
        "conservative_aggregation": app.MODES["ConExtBiMamba-Conservative"].aggregation == "mean_log_survival",
        "benchmark_table_present": table.is_file(),
        "long_demo_present": long_audio.is_file(),
        "short_demo_present": short_audio.is_file(),
    }
    detail: dict[str, Any] = {}
    if smoke:
        default = app.run_detection(long_audio, "ConExtBiMamba-Default")
        diagnostic = app.run_detection(short_audio, "ConExtBiMamba-Conservative")
        default_summary = default["summary"]
        diagnostic_summary = diagnostic["summary"]
        checks.update(
            {
                "smoke_default_three_segments": int(default_summary["n_segments"]) == 3,
                "smoke_default_benchmark_equivalent": bool(default_summary["benchmark_equivalent"]),
                "smoke_default_first_121_seconds": float(default_summary["evaluated_duration_sec"]) == 121.0,
                "smoke_default_checkpoint_hash": default_summary["checkpoint_sha256"] == EXPECTED_CHECKPOINT_HASH,
                "smoke_default_no_gain_normalization": default_summary["waveform_gain_normalization"] == "none",
                "smoke_diagnostic_nonbenchmark": not bool(diagnostic_summary["benchmark_equivalent"]),
                "smoke_diagnostic_scope": diagnostic_summary["evaluation_scope"]
                == "nonbenchmark_short_file_diagnostic",
                "smoke_csv_export": len(app.segments_with_report_metadata(default)) == 3,
                "smoke_json_export": len(app.report_payload(default)["segments"]) == 3,
            }
        )
        detail = {
            "default": {
                key: default_summary[key]
                for key in [
                    "score",
                    "threshold",
                    "prediction",
                    "n_segments",
                    "benchmark_equivalent",
                    "evaluation_scope",
                    "checkpoint_sha256",
                    "config_hash",
                    "aggregation_config_hash",
                    "device",
                ]
            },
            "short_diagnostic": {
                key: diagnostic_summary[key]
                for key in [
                    "score",
                    "threshold",
                    "prediction",
                    "n_segments",
                    "benchmark_equivalent",
                    "evaluation_scope",
                    "device",
                ]
            },
        }
    return checks, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the deployable ConExtBiMamba Windows GUI package.")
    parser.add_argument("--quick", action="store_true", help="Verify integrity and imports without model inference.")
    parser.add_argument("--smoke", action="store_true", help="Also run benchmark and short-file inference smoke tests.")
    args = parser.parse_args()
    smoke = bool(args.smoke and not args.quick)

    manifest_checks, failures = verify_manifest()
    contract_checks: dict[str, bool] = {}
    smoke_detail: dict[str, Any] = {}
    try:
        app = load_app()
        contract_checks, smoke_detail = verify_contract(app, smoke)
    except Exception as exc:
        failures.append(f"Application import or contract verification failed: {type(exc).__name__}: {exc}")

    checks = manifest_checks | contract_checks
    failed_checks = sorted(name for name, value in checks.items() if not value)
    failures.extend(f"Failed check: {name}" for name in failed_checks)
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "verified_utc": utc_now(),
        "package_root": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "mode": "smoke" if smoke else "quick",
        "checks": checks,
        "failures": failures,
        "smoke_results": smoke_detail,
    }
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "verification_latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
