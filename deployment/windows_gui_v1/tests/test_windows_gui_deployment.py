from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
DIST = HERE / "dist"
PACKAGE_NAME = "ConExtBiMamba_GUI_v1.0.0"
PACKAGE = DIST / PACKAGE_NAME
ZIP_PATH = DIST / f"{PACKAGE_NAME}_Windows.zip"
EXPECTED_CHECKPOINT_HASH = "4b17c076a4121dfe3233a28214a81b4d0a1ee1b53209e2f668b960a6bc62b00a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_required_deployment_files_present() -> None:
    required = [
        "app.py",
        "launcher.py",
        "Launch_GUI.vbs",
        "Stop_GUI.vbs",
        "Install_GUI.vbs",
        "Install_GUI_CPU.vbs",
        "Verify_GUI.vbs",
        "README_DEPLOYMENT.md",
        "DEPLOYMENT_PROVENANCE.json",
        "MANIFEST_SHA256.csv",
        "MANIFEST_SHA256.csv.sha256",
        "iara_conextbimamba/model.py",
        "iara_conextbimamba/features.py",
        "models/matched_conextbimamba_seed_20260825.pt",
        "assets/corrected_benchmark.csv",
        "examples/wavs/demo_ship_300s.wav",
        "examples/wavs/smoke_test_5s.wav",
    ]
    assert all((PACKAGE / relative).is_file() for relative in required)


def test_only_one_checkpoint_is_packaged_and_hash_locked() -> None:
    checkpoints = list(PACKAGE.rglob("*.pt"))
    assert len(checkpoints) == 1
    assert sha256(checkpoints[0]) == EXPECTED_CHECKPOINT_HASH


def test_no_dataset_or_external_evaluation_payload_is_packaged() -> None:
    paths = [path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*") if path.is_file()]
    forbidden = [
        path
        for path in paths
        if path.startswith("data/")
        or path.startswith("revision_external/")
        or path.startswith("revision_generalization/")
        or "sealed_predictions" in path
        or "_feature_cache" in path
    ]
    assert forbidden == []


def test_manifest_sidecar_and_all_rows_match() -> None:
    manifest = PACKAGE / "MANIFEST_SHA256.csv"
    sidecar_hash = (PACKAGE / "MANIFEST_SHA256.csv.sha256").read_text(encoding="utf-8").split()[0]
    assert sha256(manifest) == sidecar_hash
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 20
    for row in rows:
        path = PACKAGE / row["relative_path"]
        assert path.is_file()
        assert path.stat().st_size == int(row["size_bytes"])
        assert sha256(path) == row["sha256"]


def test_provenance_preserves_frozen_contract_and_external_warning() -> None:
    provenance = json.loads((PACKAGE / "DEPLOYMENT_PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["checkpoint_sha256"] == EXPECTED_CHECKPOINT_HASH
    assert provenance["scientific_contract"]["fixed_segment_starts"] == [162666, 808000, 1453333]
    assert provenance["scientific_contract"]["default_aggregation"] == "mean"
    assert provenance["scientific_contract"]["waveform_gain_normalization"] == "none"
    assert any("Lake Balaton specificity was 0.0" in item for item in provenance["limitations"])


def test_requested_top_level_release_banner_is_absent() -> None:
    app = (PACKAGE / "app.py").read_text(encoding="utf-8")
    assert "Offline research decision-support only — not a safety system." not in app
    assert "render_release_disclaimer" not in app


def test_console_free_launch_and_stop_contract() -> None:
    launch = (PACKAGE / "Launch_GUI.vbs").read_text(encoding="utf-8")
    stop = (PACKAGE / "Stop_GUI.vbs").read_text(encoding="utf-8")
    launcher = (PACKAGE / "launcher.py").read_text(encoding="utf-8")
    stop_ps = (PACKAGE / "Stop_GUI.ps1").read_text(encoding="utf-8")
    assert "pythonw.exe" in launch
    assert "shell.Run command, 0, False" in launch
    assert "Install_GUI.vbs" in launch
    assert "shell.Run(command, 0, True)" in launch
    assert "shell.Run(command, 0, True)" in stop
    assert '"server_address": "127.0.0.1"' in launcher
    assert '"server_headless": True' in launcher
    assert "gui_server_state.json" in launcher
    assert "os.startfile(url)" in launcher
    assert "webbrowser.open(url, new=2)" in launcher
    assert "Refusing to stop PID" in stop_ps
    assert "ExpectedLauncher" in stop_ps


def test_installer_is_local_hidden_and_official_runtime_pinned() -> None:
    auto = (PACKAGE / "Install_GUI.vbs").read_text(encoding="utf-8")
    cpu = (PACKAGE / "Install_GUI_CPU.vbs").read_text(encoding="utf-8")
    installer = (PACKAGE / "Install_GUI.ps1").read_text(encoding="utf-8")
    assert "shell.Run(command, 0, True)" in auto
    assert "shell.Run(command, 0, True)" in cpu
    assert "torch==2.12.1 torchaudio==2.11.0" in installer
    assert "https://download.pytorch.org/whl/cu130" in installer
    assert "https://download.pytorch.org/whl/cpu" in installer
    assert ".gui-venv" in installer
    assert "Python 3.12" in installer


def test_package_quick_verifier_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(PACKAGE / "scripts/verify_deployment.py"), "--quick"],
        cwd=PACKAGE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    payload = json.loads((PACKAGE / "logs/verification_latest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"


def test_package_full_inference_smoke_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(PACKAGE / "scripts/verify_deployment.py"), "--smoke"],
        cwd=PACKAGE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    payload = json.loads((PACKAGE / "logs/verification_latest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["checks"]["smoke_default_three_segments"]
    assert payload["checks"]["smoke_diagnostic_nonbenchmark"]


def test_zip_hash_crc_contents_and_no_runtime_cache() -> None:
    assert ZIP_PATH.is_file()
    sidecar = ZIP_PATH.with_suffix(ZIP_PATH.suffix + ".sha256")
    assert sha256(ZIP_PATH) == sidecar.read_text(encoding="utf-8").split()[0]
    with zipfile.ZipFile(ZIP_PATH) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
    assert len(names) == len(set(names))
    assert all(name.startswith(f"{PACKAGE_NAME}/") for name in names)
    assert not any(
        any(part in {".gui-venv", "__pycache__", ".pytest_cache", "gui_uploads", "logs"} for part in Path(name).parts)
        or name.lower().endswith((".pyc", ".pth"))
        for name in names
    )
