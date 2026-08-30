# ConExtBiMamba Windows GUI deployment QA report

## Scope

The frozen Task 7 Streamlit inference engine was packaged as a local Windows GUI with a private Python environment installer, automatic CUDA 13.0 or CPU runtime selection, console-free launcher, process-safe stop utility, package verifier, controlled-file manifest, and ZIP delivery.

No checkpoint, segment start, frontend, aggregation, threshold, score, or scientific evaluation output was changed for deployment. The packaged application uses a shorter internal checkpoint path only to avoid the Windows legacy path-length limit; the checkpoint bytes and SHA-256 remain unchanged.

## Frozen contract validation

The existing Task 7 GUI validator passed all ten checks on the real checkpoint and demonstration audio:

- checkpoint SHA-256;
- exact fixed segment starts;
- exactly three benchmark segments;
- first-121-second evaluation;
- explicit nonbenchmark short-file status;
- explicit short-file evidence scope;
- no waveform gain normalization;
- malformed-input rejection;
- CSV export helper;
- JSON export helper.

## Deployment test suite

Eleven of eleven deployment tests passed:

- required deployment files;
- exactly one checkpoint with the frozen hash;
- no packaged datasets or external-evaluation payloads;
- complete controlled-file manifest and sidecar;
- frozen contract and external-warning provenance;
- requested front-page banner absence;
- console-free launcher and process-safe stop contract;
- hidden local installer with pinned official PyTorch runtimes;
- quick package verifier;
- real CUDA inference smoke test;
- ZIP hash, CRC, duplicate-path, and cache-exclusion checks.

The CUDA smoke test produced a benchmark-equivalent three-segment result for the bundled 300-second example and an explicitly nonbenchmark result for the five-second smoke file. Both used the frozen seed-20260825 checkpoint with SHA-256 `4b17c076a4121dfe3233a28214a81b4d0a1ee1b53209e2f668b960a6bc62b00a`.

## Browser and launcher QA

The packaged launcher was started with `pythonw.exe`, a hidden window, and redirected logs. `MainWindowHandle` was `0` immediately after launch and again after the health endpoint returned HTTP 200. The server bound only to `127.0.0.1`.

Interactive browser checks confirmed:

- default-mode sidebar, arithmetic-mean aggregation, threshold, and short model path;
- bundled-example selection;
- successful three-segment inference and decision card;
- score, threshold, duration, and top-segment metrics;
- waveform panel and CSV/JSON/PNG download controls;
- corrected matched-benchmark comparison panel;
- About panel with model, mode, version, reproducibility, input-contract, and upload-cache information;
- confirmed absence of the removed Citation And License and Known Limitations sections;
- confirmed absence of the removed benchmark explanatory caption;
- switching to the conservative frozen operating mode;
- no browser console warnings or errors.

The packaged stop utility verified the exact recorded launcher command line, terminated only that process, removed the state file, and left no GUI process running.

## Safety and evidence limitations

The dashboard is offline research decision-support, not a real-time detector, safety system, vessel census, or operational alerting product. It binds to localhost only. Uploaded audio is cached locally under `runs/gui_uploads` until the user deletes it.

The packaged documentation records the external-evaluation boundaries and domain-shift limitations. The rendered interface contains the requested concise About and Benchmark panels, and the user-requested top-level warning banner is not displayed on the Analyze WAV front page.
