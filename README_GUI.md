# ConExtBiMamba Windows GUI

The Windows GUI is a local Streamlit dashboard for acoustic ship/background
screening. It supports single-file inference, batch processing, waveform and
spectrogram review, segment-level probabilities, recording-level decisions, and
reproducible CSV, JSON, PNG, and ZIP exports.

The application binds to `127.0.0.1`; uploaded audio and inference results remain
on the local computer unless the user exports or moves them.

## Download

Use the verified portable package:

[`deployment/windows_gui_v1/dist/ConExtBiMamba_GUI_v1.0.0_Windows.zip`](deployment/windows_gui_v1/dist/ConExtBiMamba_GUI_v1.0.0_Windows.zip)

The release also includes a walkthrough video:

[`media/Mamba_ship_detection.mp4`](media/Mamba_ship_detection.mp4)

The same video is bundled in the portable package. In the GUI's **About** tab it
autoplays muted; viewers can enable audio from the standard player controls.

## Windows installation

1. Extract the complete ZIP to a normal writable folder.
2. Double-click `Install_GUI.vbs`. The installer selects the CUDA 13.0 runtime
   when it detects a compatible NVIDIA GPU and otherwise uses the CPU runtime.
3. If automatic GPU installation is unsuitable, use `Install_GUI_CPU.vbs`.
4. Double-click `Launch_GUI.vbs` to start the hidden local server and open the
   dashboard.
5. Use `Verify_GUI.vbs` to check the package, checkpoint, imports, and frozen
   inference contract.
6. Use `Stop_GUI.vbs` to stop only this installation's recorded server process.

No administrator rights are required. Installation and launch logs are written
inside the extracted package rather than displayed in a console window.

## Frozen model contract

The package contains one representative matched ConExtBiMamba checkpoint:

- Seed: `20260825`
- Checkpoint SHA-256:
  `4b17c076a4121dfe3233a28214a81b4d0a1ee1b53209e2f668b960a6bc62b00a`
- Sample rate: 16 kHz
- Benchmark bag length: 121 seconds
- Segment length: 20 seconds
- Fixed segment starts: samples `162666`, `808000`, and `1453333`
- Waveform gain normalization: none
- Frontend normalization: per-segment CMVN after the frozen frontend

The GUI accepts shorter recordings as nonbenchmark diagnostics, but a
benchmark-equivalent decision requires at least 121 seconds of audio.

## Operating modes

| Mode | Aggregation | Threshold | Role |
|---|---|---:|---|
| `ConExtBiMamba-Default` | Arithmetic mean | `0.2287546408673127` | Frozen primary offline research mode |
| `ConExtBiMamba-Conservative` | Mean log survival | `0.6510023052576646` | Frozen secondary mode; not an operational guarantee |

Both modes use the same checkpoint, frontend, and fixed evidence windows. Their
thresholds were frozen before the external Taitung and Lake Balaton evaluations.

## Accepted inputs

- WAV, FLAC, AIFF/AIF, OGG, and MP3 when the local audio stack supports the
  codec
- Mono or multichannel audio; channels are averaged to mono
- Any input sample rate; audio is resampled to 16 kHz
- Single files or local batches

For a benchmark-equivalent 121-second recording, the GUI evaluates exactly the
three frozen segments. Longer inputs are limited to the first 121 seconds under
that contract. Shorter inputs are explicitly marked as diagnostic.

## Outputs

Single-file analysis provides:

- recording score, threshold, prediction, and uncertainty flag;
- waveform and spectrogram views;
- segment-probability timeline;
- event summary table;
- segment CSV;
- decision JSON; and
- analysis PNG.

Batch analysis provides:

- one recording-level row per input;
- per-file status and error information;
- batch CSV and JSON; and
- a ZIP containing summaries, segment tables, and event tables.

Exports include the selected mode, aggregation method, threshold, checkpoint
path and SHA-256, configuration hashes, device, frontend description, and
PyTorch version.

## Package integrity

The extracted package contains:

- `MANIFEST_SHA256.csv`, listing every controlled file and checksum;
- `MANIFEST_SHA256.csv.sha256`, authenticating the manifest;
- `models/checkpoint.sha256`, authenticating the frozen checkpoint; and
- `scripts/verify_deployment.py`, used by `Verify_GUI.vbs`.

The Windows deployment test suite checks required files, checkpoint identity,
absence of datasets and caches, package-manifest integrity, hidden launch/stop
behavior, real inference, ZIP contents, duplicate paths, and CRC integrity.

## Repository development

The repository root contains the editable Streamlit source in `app.py`, the
model implementation in `iara_conextbimamba/`, and the reproducible package
builder in `deployment/windows_gui_v1/build_windows_gui.py`.

To run the development app from a configured checkout:

```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1 --server.fileWatcherType none
```

The development app expects the frozen checkpoint at the repository path encoded
in `app.py`. Users who only need inference should use the portable Windows ZIP,
which contains the checkpoint at a short internal path.

## Data and use boundaries

The package does not contain the full IARA dataset, Taitung data, Lake Balaton
data, training caches, private virtual environments, or sealed-evaluation
predictions. Obtain IARA from its official archive and follow the applicable
dataset terms.

This software is an offline research interface, not a safety-certified
navigation or collision-avoidance system. Predictions should be interpreted by
qualified users in the context of the recording environment and the complete
evaluation evidence.

See `LICENSE`, `MODEL_CARD.md`, and `CITATION.cff` for the software license,
model scope, and citation metadata.
