# ConExtBiMamba Ship Detector — Windows GUI 1.0

This package is a local, offline research dashboard for acoustic ship/background screening. Audio inference runs on this computer; the Streamlit server binds only to `127.0.0.1` and is not exposed to the local network or internet.

It is not a safety system, an autonomous alarm, an AIS-verified vessel detector, or an operational false-alarm-rate monitor. External evaluations found severe background-specificity loss under domain shift. Every positive result requires qualified human review.

## First installation

Requirements:

- Windows 10 or 11, 64 bit.
- Python 3.12, 64 bit, available through the Windows `py` launcher or `python` command.
- Internet access during first installation to download the Python runtime packages.
- Approximately 5 GB of free disk space for the CUDA runtime, or approximately 1.5 GB for CPU-only installation.

Recommended installation:

1. Extract the complete ZIP to a normal writable folder. Do not run it from inside the ZIP.
2. Double-click `Install_GUI.vbs`. It installs CUDA 13.0 packages when an NVIDIA GPU is detected and otherwise installs CPU-only packages. No administrator rights are required. You may instead double-click `Launch_GUI.vbs`; when the private runtime is absent, it offers to run the installer and then launches automatically after a successful installation.
3. If the automatic GPU installation fails because of an older driver or unsupported hardware, double-click `Install_GUI_CPU.vbs`.
4. Installation runs without a console window. Progress and errors are recorded in `logs/install.log`; a completion dialog is shown.

The validated CUDA environment uses PyTorch 2.12.1 with CUDA 13.0 and TorchAudio 2.11.0. TorchAudio 2.11 uses the stable PyTorch ABI and supports PyTorch 2.11 and later. CPU installation uses the corresponding official CPU wheel index.

## Launch and stop

- Double-click `Launch_GUI.vbs` to start the local server without a console window and open the dashboard through the native Windows URL handler. Python's default-browser mechanism is retained as a fallback.
- Double-click `Stop_GUI.vbs` to stop only the server process recorded by this installation.
- Double-click `Verify_GUI.vbs` to recheck package integrity, imports, checkpoint identity, and the frozen inference contract.

If the browser does not open automatically, inspect `runs/gui_server_state.json` and open its `url` value. The usual address is `http://127.0.0.1:8501`.

## Input contract

Benchmark-equivalent evaluation requires at least 121 seconds of audio. The application:

- converts supported input to mono;
- resamples to 16 kHz;
- applies no waveform gain normalization;
- evaluates the first 121 seconds;
- uses exactly three 20-second segments beginning at samples 162,666, 808,000, and 1,453,333;
- applies the frozen multi-resolution frontend and per-segment CMVN;
- produces the default recording score as the arithmetic mean of the three probabilities.

Shorter files are accepted only as prominently labeled nonbenchmark diagnostics.

## Frozen modes

- `ConExtBiMamba-Default`: arithmetic mean, threshold `0.2287546408673127`.
- `ConExtBiMamba-Conservative`: mean-log-survival aggregation, threshold `0.6510023052576646`. The name does not imply an operational guarantee.

Neither threshold was chosen from Lake Balaton, Taitung, or other external-test outcomes.

## Outputs and local data

The GUI exports segment CSV, decision JSON, analysis PNG, batch CSV/JSON, and batch ZIP files. Uploaded audio is cached locally under `runs/gui_uploads`. Delete that folder after use if retained copies are not wanted.

The package contains two demonstration recordings only: one 300-second curated example and one 5-second nonbenchmark smoke file. It does not contain the IARA dataset, Lake Balaton data, Taitung data, training caches, or sealed-evaluation predictions.

## Known limitations

- The frozen model is research decision-support, not a causal real-time detector.
- Performance can change with hydrophone, site, vessel class, range, propagation, weather, biological noise, and recording hardware.
- On the small sealed Lake Balaton evaluation, the default model detected all five metadata-supported ship bags but rejected none of five background bags at every seed: sensitivity 1.0 and specificity 0.0.
- On the candidate-conditioned sealed Taitung evaluation, sensitivity was high but background rejection was poor. That result is not a monitoring-hour false-alarm rate.
- Lake Balaton and Taitung acoustic labels are not AIS-confirmed vessel truth.

## Integrity

`MANIFEST_SHA256.csv` lists the expected size and SHA-256 of every controlled package file. `MANIFEST_SHA256.csv.sha256` authenticates that manifest. The frozen checkpoint must have SHA-256:

`4b17c076a4121dfe3233a28214a81b4d0a1ee1b53209e2f668b960a6bc62b00a`

## License and citation

See `LICENSE`, `MODEL_CARD.md`, and `CITATION.cff`. The package is intended for the same research and citation context as the accompanying project. Confirm all dataset and deployment permissions before redistribution.
