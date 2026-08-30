# Windows GUI deployment builder

This directory builds the portable Windows distribution of the frozen ConExtBiMamba Streamlit dashboard. It copies only the application, inference modules, one frozen checkpoint, the corrected benchmark summary, two curated demonstration WAV files, release metadata, and deployment utilities.

The build deliberately excludes virtual environments, datasets, training runs, evaluation caches, unrelated revision outputs, and all nonrepresentative checkpoints.

Build from the repository root with:

```powershell
.\.venv\Scripts\python.exe deployment\windows_gui_v1\build_windows_gui.py
```

Run the deployment tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest deployment\windows_gui_v1\tests -v
```

The generated folder and ZIP are written under `deployment/windows_gui_v1/dist/`.

