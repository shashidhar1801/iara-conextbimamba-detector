# Public Release Notes

This package is prepared as a manuscript-supporting research release for the Underwater ConExtBiMamba Detector GUI and inference code.

## Included

- Streamlit GUI: `app.py`.
- GUI/user documentation: `README_GUI.md`.
- Pinned public-release dependency file: `requirements.txt`.
- Software license: `LICENSE`.
- Citation metadata: `CITATION.cff`.
- Model card: `MODEL_CARD.md`.
- Model/source package: `iara_conextbimamba/`.
- Training and inference scripts: `scripts/`.
- Model configs: `configs/`.
- Final proposed checkpoint: `runs/iara_conextbimamba_full_large_multires_20s/best.pt`.
- Optional research baseline checkpoints: ResNet and Conformer.
- Benchmark metric files and aggregation-threshold summaries used by the GUI.
- Supplementary GUI figure: `docs/Supplementary_Figure_S9_GUI_Dashboard.tiff`.

## Excluded

- Local virtual environments such as `.venv/`.
- Python bytecode caches such as `__pycache__/`.
- Streamlit/runtime upload caches such as `runs/gui_uploads/`.
- Raw IARA audio archives and extracted IARA WAV files.
- Local scratch files used during development.

IARA audio should be obtained directly from the official IARA/Zenodo release and used under its dataset license.

## Recommended Smoke Test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.maxUploadSize 1024 --server.fileWatcherType none
```

Then test one short WAV, one longer WAV, one batch, Balanced mode, Conservative mode, advanced research mode, missing-checkpoint behavior, missing-example behavior, and CSV/JSON/PNG/ZIP exports.
