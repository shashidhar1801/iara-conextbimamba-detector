from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE / "source"
DIST = HERE / "dist"
PACKAGE_NAME = "ConExtBiMamba_GUI_v1.0.0"
PACKAGE = DIST / PACKAGE_NAME
ZIP_PATH = DIST / f"{PACKAGE_NAME}_Windows.zip"
SOURCE_CHECKPOINT_REL = Path(
    "revision_training/task3_matched_models/frozen_checkpoints/"
    "matched_conextbimamba/seed_20260825/best_checkpoint.pt"
)
PACKAGE_CHECKPOINT_REL = Path("models/matched_conextbimamba_seed_20260825.pt")
CHECKPOINT_SHA = "4b17c076a4121dfe3233a28214a81b4d0a1ee1b53209e2f668b960a6bc62b00a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(relative: str | Path) -> None:
    relative = Path(relative)
    source = REPO / relative
    target = PACKAGE / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_file_as(source_relative: str | Path, target_relative: str | Path) -> None:
    source = REPO / Path(source_relative)
    target = PACKAGE / Path(target_relative)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def controlled_files() -> list[Path]:
    excluded_parts = {".gui-venv", "logs", "runs", "__pycache__", ".pytest_cache"}
    excluded_names = {"MANIFEST_SHA256.csv", "MANIFEST_SHA256.csv.sha256"}
    return sorted(
        path
        for path in PACKAGE.rglob("*")
        if path.is_file()
        and path.name not in excluded_names
        and not any(part in excluded_parts for part in path.relative_to(PACKAGE).parts)
        and path.suffix.lower() not in {".pyc"}
    )


def main() -> None:
    dist_resolved = DIST.resolve()
    package_resolved = PACKAGE.resolve()
    if package_resolved.parent != dist_resolved or DIST.parent != HERE:
        raise RuntimeError("Refusing to rebuild outside deployment/windows_gui_v1/dist")
    DIST.mkdir(parents=True, exist_ok=True)
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    sidecar = ZIP_PATH.with_suffix(ZIP_PATH.suffix + ".sha256")
    if sidecar.exists():
        sidecar.unlink()

    shutil.copytree(SOURCE, PACKAGE, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative in ["app.py", "LICENSE", "MODEL_CARD.md", "CITATION.cff"]:
        copy_file(relative)
    for source in sorted((REPO / "iara_conextbimamba").glob("*.py")):
        copy_file(source.relative_to(REPO))
    copy_file_as(SOURCE_CHECKPOINT_REL, PACKAGE_CHECKPOINT_REL)
    copy_file_as(SOURCE_CHECKPOINT_REL.with_name("best_checkpoint.sha256"), "models/checkpoint.sha256")
    copy_file_as(SOURCE_CHECKPOINT_REL.with_name("validation_threshold.json"), "models/validation_threshold.json")
    copy_file_as(
        "revision_manuscript/task7_final_revision/tables/Main_Table_2_Corrected_Matched_Benchmark.csv",
        "assets/corrected_benchmark.csv",
    )
    copy_file_as(
        "examples/wavs/01_01_clear_ship_close_cpa_large_tanker__A-0420_4ef096f21fb3.wav",
        "examples/wavs/demo_ship_300s.wav",
    )
    copy_file_as("examples/wavs/smoke_test_5s.wav", "examples/wavs/smoke_test_5s.wav")
    copy_file_as("docs/Supplementary_Figure_S9_GUI_Dashboard.png", "assets/gui_preview.png")
    copy_file_as("media/Mamba_ship_detection.mp4", "assets/Mamba_ship_detection.mp4")

    deployed_app = PACKAGE / "app.py"
    app_text = deployed_app.read_text(encoding="utf-8")
    old_checkpoint_block = '''FROZEN_CHECKPOINT = (
    "revision_training/task3_matched_models/frozen_checkpoints/"
    "matched_conextbimamba/seed_20260825/best_checkpoint.pt"
)'''
    new_checkpoint_block = 'FROZEN_CHECKPOINT = "models/matched_conextbimamba_seed_20260825.pt"'
    if app_text.count(old_checkpoint_block) != 1:
        raise RuntimeError("Could not deterministically map the repository checkpoint path to the short deployment path")
    app_text = app_text.replace(old_checkpoint_block, new_checkpoint_block)
    old_benchmark_path = (
        'PROJECT_ROOT / "revision_manuscript/task7_final_revision/tables/'
        'Main_Table_2_Corrected_Matched_Benchmark.csv"'
    )
    new_benchmark_path = 'PROJECT_ROOT / "assets/corrected_benchmark.csv"'
    if app_text.count(old_benchmark_path) != 1:
        raise RuntimeError("Could not deterministically map the corrected benchmark table to the short deployment path")
    deployed_app.write_text(app_text.replace(old_benchmark_path, new_benchmark_path), encoding="utf-8")

    copied_checkpoint = PACKAGE / PACKAGE_CHECKPOINT_REL
    if sha256(copied_checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError("Frozen checkpoint hash changed during deployment build")

    source_commit = git("rev-parse", "HEAD")
    source_branch = git("branch", "--show-current")
    provenance = {
        "deployment": "ConExtBiMamba Windows GUI",
        "deployment_version": "1.0.0",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "source_branch": source_branch,
        "source_commit": source_commit,
        "app_sha256": sha256(PACKAGE / "app.py"),
        "source_checkpoint_relative_path": SOURCE_CHECKPOINT_REL.as_posix(),
        "checkpoint_relative_path": PACKAGE_CHECKPOINT_REL.as_posix(),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "representative_seed": 20260825,
        "validated_runtime": {
            "python": "3.12.10",
            "torch": "2.12.1+cu130",
            "torchaudio": "2.11.0+cu130",
            "streamlit": "1.58.0",
            "cuda": "13.0",
        },
        "scientific_contract": {
            "sample_rate": 16000,
            "bag_samples": 1936000,
            "segment_samples": 320000,
            "fixed_segment_starts": [162666, 808000, 1453333],
            "default_aggregation": "mean",
            "default_threshold": 0.2287546408673127,
            "waveform_gain_normalization": "none",
        },
        "scope": "offline local research decision-support",
        "limitations": [
            "not safety certified",
            "not an AIS-verified vessel detector",
            "not an operational monitoring-hour false-alarm evaluation",
            "Lake Balaton specificity was 0.0 on five frozen background bags at every seed",
            "qualified human review is mandatory",
        ],
        "official_runtime_sources": {
            "pytorch_previous_versions": "https://pytorch.org/get-started/previous-versions/",
            "torchaudio_installation": "https://docs.pytorch.org/audio/main/installation.html",
        },
    }
    (PACKAGE / "DEPLOYMENT_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    inventory = []
    for path in controlled_files():
        inventory.append(
            {
                "relative_path": path.relative_to(PACKAGE).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = PACKAGE / "MANIFEST_SHA256.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(inventory)
    manifest_hash = sha256(manifest)
    (PACKAGE / "MANIFEST_SHA256.csv.sha256").write_text(
        f"{manifest_hash}  MANIFEST_SHA256.csv\n", encoding="utf-8"
    )

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file():
                archive.write(path, (Path(PACKAGE_NAME) / path.relative_to(PACKAGE)).as_posix())
    with zipfile.ZipFile(ZIP_PATH) as archive:
        bad = archive.testzip()
        names = archive.namelist()
    if bad is not None:
        raise RuntimeError(f"ZIP CRC failure at {bad}")
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate paths were written to the deployment ZIP")
    prohibited = [
        name
        for name in names
        if any(part in {".gui-venv", "__pycache__", ".pytest_cache", "gui_uploads"} for part in Path(name).parts)
        or name.lower().endswith((".pyc", ".pth"))
    ]
    if prohibited:
        raise RuntimeError(f"Prohibited deployment ZIP content: {prohibited}")

    zip_hash = sha256(ZIP_PATH)
    sidecar.write_text(f"{zip_hash}  {ZIP_PATH.name}\n", encoding="utf-8")
    report = {
        "status": "BUILT_PENDING_TESTS",
        "built_utc": provenance["built_utc"],
        "package_root": str(PACKAGE),
        "package_name": PACKAGE_NAME,
        "controlled_files": len(inventory),
        "manifest_sha256": manifest_hash,
        "zip_path": str(ZIP_PATH),
        "zip_size_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": zip_hash,
        "zip_entries": len(names),
        "zip_crc_passed": True,
        "duplicate_zip_paths": 0,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "source_commit": source_commit,
    }
    (DIST / "DEPLOYMENT_BUILD_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
