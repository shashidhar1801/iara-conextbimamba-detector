# Underwater ConExtBiMamba Streamlit GUI

This dashboard is a research decision-support interface for passive underwater acoustic ship/background detection from audio recordings. It wraps the trained IARA benchmark models and provides single-file analysis, batch analysis, visualization, recording-level decisions, segment/event summaries, and reproducible exports.

The GUI is not safety-certified and must not be used as the sole basis for operational marine monitoring decisions.

## Installation

Use Python 3.10 or newer. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.fileWatcherType none
```

On Windows, activate the environment with `.venv\Scripts\activate`.

For larger local uploads, raise Streamlit's upload cap:

```bash
streamlit run app.py --server.maxUploadSize 1024 --server.fileWatcherType none
```

## Public Release Smoke Test

Before publishing a release package, run the dashboard in a clean environment and test:

- One short WAV.
- One long WAV.
- One batch of WAV files.
- `ConExtBiMamba-Balanced`.
- `ConExtBiMamba-Conservative`.
- Advanced research mode.
- Missing checkpoint behavior.
- Missing bundled example WAV behavior.
- CSV, JSON, and PNG export.
- Batch ZIP export.

## Recommended Public Release Layout

The GUI ZIP by itself is not standalone because the app expects the model source, configs, and checkpoints to exist beside it. For public release, package the dashboard with the full project structure:

```text
project_root/
  app.py
  README_GUI.md
  requirements.txt
  LICENSE
  CITATION.cff
  MODEL_CARD.md
  iara_conextbimamba/
  configs/
  runs/
    iara_conextbimamba_full_large_multires_20s/best.pt
    iara_resnet_full_fast/best.pt        optional research baseline
    iara_conformer_full_fast/best.pt     optional research baseline
  examples/
    wavs/                                small bundled demo WAV files
  docs/
    Supplementary_Figure_S9_GUI_Dashboard.tiff
```

Do not include local upload caches, virtual environments, Python bytecode caches, or the full IARA audio archive in a public release. The full dataset is already public at https://doi.org/10.5281/zenodo.15758636.

## Required Files

The GUI expects this app to live beside the trained project files:

- `app.py`: Streamlit application.
- `iara_conextbimamba/`: model and feature-extraction source code.
- `runs/iara_conextbimamba_full_large_multires_20s/best.pt`: trained ConExtBiMamba checkpoint.
- `runs/iara_resnet_full_fast/best.pt`: optional ResNet research baseline checkpoint.
- `runs/iara_conformer_full_fast/best.pt`: optional Conformer research baseline checkpoint.
- `runs/*/eval_test/metrics.json`: optional benchmark metrics for the comparison tab.
- `examples/wavs/`: small bundled WAV files for dashboard testing.
- `data/iara_full/`: optional local full IARA audio folder. The GUI uses this only when present locally.
- `LICENSE`, `CITATION.cff`, and `MODEL_CARD.md`: public-release metadata.
- `docs/Supplementary_Figure_S9_GUI_Dashboard.tiff`: manuscript-ready GUI screenshot.

If a selected checkpoint is missing, the dashboard shows a friendly validation message before inference.

## Model Modes

- `ConExtBiMamba-Balanced`: primary public mode. Uses top-2 mean aggregation with threshold `0.9298714101`.
- `ConExtBiMamba-Conservative`: secondary public mode for low false-alarm review. Uses noisy-or evidence aggregation with threshold `34.5572580033`.
- `ConExtBiMamba mean (research)`: research/reference mean aggregation for the proposed detector.
- `ResNet baseline (research)`: spectrogram CNN baseline.
- `Conformer baseline (research)`: MHSA Conformer baseline.

Only Balanced and Conservative are shown by default. Research modes are available from the advanced sidebar toggle.

## Input Requirements

- WAV and FLAC are recommended. AIFF, OGG, and MP3 may work when supported by the local audio stack.
- Mono or stereo inputs are accepted; channels are averaged to mono.
- Audio is resampled to the checkpoint sample rate, typically 16 kHz.
- The final ConExtBiMamba GUI model uses the checkpoint segmentation settings, typically 20 s segments with the stored hop size.
- Very long files are allowed locally, but files longer than 30 minutes or larger than 500 MB trigger warnings.

## Outputs

The `Analyze WAV` tab provides:

- Recording decision card with score, threshold, prediction, uncertainty flag, and segment count.
- Waveform plot.
- Spectrogram plot.
- Segment probability timeline.
- Segment-level CSV export with model, threshold, checkpoint hash, config hash, device, and torch metadata.
- JSON export containing summary, events, and segment records.
- PNG export of the analysis figure.

The `Batch Processing` tab provides:

- Per-file status and error columns.
- Summary counts for completed, failed, and review-recommended files.
- Batch CSV and JSON exports.
- ZIP export containing batch summaries, segment CSV files, and event CSV files.

## Event And Uncertainty Logic

Recordings near the operating threshold are marked with `review` in the `uncertainty_flag` field. Consecutive segments with ship probability at or above `0.5` are merged into acoustic events with start time, end time, duration, segment count, max probability, and mean probability.

## Known Limitations

This is a research prototype. Performance can change across hydrophones, sites, vessel types, propagation conditions, sea state, biological noise, and recording hardware. Human review remains important, especially for borderline outputs or unfamiliar deployment environments.

## Citation

Cite the IARA dataset according to the Zenodo/IARA release. Cite the IARA benchmark article for the published MLP/RF/CNN baseline context. For the architecture rationale, cite:

Zhang et al., "Mamba in Speech: Towards an Alternative to Self-Attention," IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2025.

## License And Data Policy

Do not redistribute IARA audio unless the dataset license permits it. Keep large checkpoints and audio outside public repositories when licensing, file size, or privacy constraints require it.
