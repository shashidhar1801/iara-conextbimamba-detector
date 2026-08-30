# Model Card: Underwater ConExtBiMamba Detector

## Summary

The Underwater ConExtBiMamba Detector is a research model for passive underwater acoustic ship/background detection. It uses a Conformer-style acoustic encoder in which multi-head self-attention is replaced by external bidirectional Mamba, while retaining feed-forward, convolutional, residual, and normalization components. The public Streamlit GUI exposes two recommended operating modes:

- `ConExtBiMamba-Balanced`: primary balanced/F1 operating mode using top-2 mean aggregation.
- `ConExtBiMamba-Conservative`: low false-alarm monitoring mode using noisy-or evidence aggregation.

Advanced research modes include mean aggregation for ConExtBiMamba plus ResNet and Conformer baselines.

## Intended Use

This package is intended for:

- Research on passive acoustic ship-presence prediction.
- Reproducible manuscript review of the IARA benchmark experiments.
- Exploratory marine acoustic decision support where predictions are reviewed by a qualified human user.
- Demonstrating recording-level and segment-level model evidence through the Streamlit dashboard.

## Out-of-Scope Use

Do not use this model as the sole basis for:

- Safety-critical marine navigation or collision-avoidance decisions.
- Legal, enforcement, military, or regulatory decisions.
- Fully automated alerting without site-specific validation and human review.
- Claims of universal vessel detection across sensors, oceans, seasons, or vessel classes.

## Training Dataset

The final GUI model was trained and evaluated using the IARA public passive-acoustic ship/background benchmark. The training setup used recording-level ship/background labels, full-dataset splits, waveform preprocessing, resampling, segmentation, multi-resolution log-Mel/CQT-style front-end features, and validation-selected recording-level aggregation thresholds.

IARA audio is not redistributed in this release package. Users should obtain IARA directly from the official Zenodo/IARA release and follow its citation and license requirements.

## Inputs

- Recommended audio: WAV or FLAC.
- Other formats: AIFF, OGG, and MP3 may work when supported by the local audio stack.
- Channel handling: mono or stereo accepted; channels are averaged to mono.
- Resampling: audio is resampled to the checkpoint sample rate, typically 16 kHz.
- Segmentation: the final model uses the checkpoint configuration, typically 20 s analysis segments.

## Outputs

The GUI and inference utilities provide:

- Recording-level prediction: `ship` or `background`.
- Recording-level score, threshold, and aggregation method.
- Segment-level ship probabilities.
- Top-evidence segment.
- Uncertainty flag for borderline recordings.
- Event summaries from consecutive ship-positive segments.
- CSV, JSON, PNG, and batch ZIP exports with model/checkpoint metadata.

## Operating Modes

| Mode | Aggregation | Purpose |
| --- | --- | --- |
| `ConExtBiMamba-Balanced` | Top-2 mean | Primary public mode; balanced/F1 operating point. |
| `ConExtBiMamba-Conservative` | Noisy-or evidence | Low false-alarm monitoring mode. |
| `ConExtBiMamba mean (research)` | Mean | Research/reference aggregation. |
| `ResNet baseline (research)` | Mean | CNN spectrogram baseline. |
| `Conformer baseline (research)` | Mean | MHSA Conformer baseline. |

## Limitations

Performance may vary with:

- Hydrophone and recorder characteristics.
- Recording site, water depth, propagation range, and bathymetry.
- Vessel type, speed, aspect, and source level.
- Weather, sea state, biological sounds, and anthropogenic background noise.
- Domain shift between IARA and a new deployment environment.

Borderline predictions should be reviewed manually. Site-specific validation is recommended before any operational use.

## Ethical And Safety Considerations

The model is a research decision-support tool. It can produce false positives and false negatives. Conservative mode reduces false alarms but may miss some ship recordings. Balanced mode improves balanced performance but may produce more false alarms in difficult background recordings. Human review and local validation remain necessary.

## Reproducibility Metadata

Exports include:

- GUI/app version.
- Model mode and aggregation method.
- Threshold.
- Checkpoint path and SHA-256 hash.
- Configuration hash.
- Sample rate, segment length, and hop length.
- Front-end description.
- Device and PyTorch version.

## Citation

If you use this model or GUI, cite:

- The associated Underwater ConExtBiMamba manuscript once available.
- IARA: An Acoustical Recordings Archive, following the official Zenodo/IARA citation.
- The IARA benchmark article for published RF/MLP/CNN context.
- Zhang et al., "Mamba in Speech: Towards an Alternative to Self-Attention," IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2025.

## License And Data Policy

The software source code is provided under the repository `LICENSE`. IARA audio is not included and must be obtained under the IARA dataset terms. Trained weights may be subject to separate release terms if required by the dataset or hosting venue.
