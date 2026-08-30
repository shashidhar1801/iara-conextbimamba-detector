# Deployable GUI completion report

The ConExtBiMamba Windows GUI 1.0 deployment is complete and verified.

## Delivery

- Folder: `deployment/windows_gui_v1/dist/ConExtBiMamba_GUI_v1.0.0/`
- ZIP: `deployment/windows_gui_v1/dist/ConExtBiMamba_GUI_v1.0.0_Windows.zip`
- ZIP size: 29,587,448 bytes.
- ZIP SHA-256: `5e9492060fa521ed9ed6869a393b8077fecfc9798fb6b881f65a92a0a211d435`.
- ZIP entries: 32.
- Controlled files: 30 plus the manifest and manifest sidecar.
- Manifest SHA-256: `055597409f5b5cb4c7ce56db71b2a9a7642e9f95d770d7aaec37e133ee50ac65`.
- ZIP CRC/full-entry read: pass.
- Duplicate paths: 0.
- Prohibited paths: 0.

## Installation and launch

After extracting the ZIP, double-click `Install_GUI.vbs`. It creates a private `.gui-venv` without administrator rights and installs the official CUDA 13.0 runtime when an NVIDIA GPU is detected, otherwise the CPU runtime. `Install_GUI_CPU.vbs` is the explicit fallback. Installation is hidden and logged to `logs/install.log`. If `Launch_GUI.vbs` is used before installation, it now offers to run the hidden installer and launches automatically after successful installation.

Double-click `Launch_GUI.vbs` to start the localhost-only dashboard without a console window. It opens the ready URL through the native Windows URL handler and retains Python's browser mechanism as a fallback. Double-click `Stop_GUI.vbs` to safely stop only the recorded package launcher process. `Verify_GUI.vbs` rechecks controlled files, checkpoint identity, dependencies, and the frozen contract.

## Verification

- Frozen Task 7 GUI contract: 10/10 passed.
- Deployment/package tests: 11/11 passed.
- Real CUDA inference smoke: passed.
- Hidden launch before readiness: `MainWindowHandle = 0`.
- Hidden launch after HTTP 200 readiness: `MainWindowHandle = 0`.
- Browser interaction and visual QA: passed.
- Browser console warnings/errors: 0.
- Requested top-level release banner absent from the rendered front page: passed.
- First-launch missing-runtime installer-handoff contract: passed.
- Native Windows URL opening with fallback: passed.
- Safe stop and state cleanup: passed.
- Final clean package contains no runtime logs, private environment, bytecode, datasets, external-evaluation predictions, or training caches.

## Scientific integrity

The package contains exactly one representative seed-20260825 checkpoint with SHA-256 `4b17c076a4121dfe3233a28214a81b4d0a1ee1b53209e2f668b960a6bc62b00a`. The checkpoint, frontend, three fixed segment starts, no-gain-normalization rule, aggregation modes, thresholds, and prior scientific results were not changed.

The repository documentation records the external-evaluation scope and domain-shift limitations. The GUI identifies itself as offline research decision-support and retains the frozen scientific contract.
