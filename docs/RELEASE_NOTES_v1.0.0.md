# ConExtBiMamba Windows GUI v1.0.0

This first public GUI release packages the frozen seed-20260825 ConExtBiMamba detector as a local Windows Streamlit application.

## Highlights

- Console-free Windows installation, launch, integrity verification, and stop utilities.
- Frozen default and conservative recording-level operating modes.
- Single-WAV and batch processing with CSV, JSON, PNG, and ZIP exports.
- Bundled checkpoint and package-level SHA-256 inventory.
- Interface walkthrough bundled in the **About** tab, with muted autoplay.
- Demonstration audio archived at [doi:10.5281/zenodo.22177090](https://doi.org/10.5281/zenodo.22177090).

## Install and launch

1. Download `ConExtBiMamba_GUI_v1.0.0_Windows.zip` from the Release assets.
2. Extract the complete ZIP to a writable folder.
3. Double-click `Verify_GUI.vbs` to verify package integrity.
4. Double-click `Install_GUI.vbs` for automatic GPU/CPU installation, or `Install_GUI_CPU.vbs` for CPU-only installation.
5. Double-click `Launch_GUI.vbs`.

The dashboard binds only to `127.0.0.1`; uploaded audio is processed locally.

## Frozen inference contract

- Sample rate: 16 kHz.
- Benchmark bag: 121 seconds.
- Evidence: three 20-second segments starting at samples 162,666, 808,000, and 1,453,333.
- Default mode: arithmetic mean, threshold `0.2287546408673127`.
- Conservative mode: mean-log-survival, threshold `0.6510023052576646`.
- Checkpoint SHA-256: `4b17c076a4121dfe3233a28214a81b4d0a1ee1b53209e2f668b960a6bc62b00a`.

## Scope

This software is an offline research interface. It is not an AIS-verified vessel detector, a vessel census, a safety system, or an operational false-alarm-per-hour monitor. External performance is deployment-dependent, and qualified interpretation remains necessary.
