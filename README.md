# NanoSleepNet-TCN

**NanoSleepNet-TCN: A Compact Temporal Model for On-Device Single-Channel EEG Sleep Staging**

NanoSleepNet-TCN is a compact PyTorch model for single-channel EEG sleep staging on resource-constrained edge devices. It compresses each 30-s EEG epoch into a 64-dimensional morphology token and performs lightweight temporal refinement in token space, including a causal token-cached mode for online inference.

> **Manuscript status:** submitted to *IEEE Transactions on Biomedical Engineering*.

## Authors

| Author | Affiliation | Email | Role |
|---|---|---|---|
| Guisong Liu | School of Biological Science and Medical Engineering, Southeast University, Nanjing, China | 230258331@seu.edu.cn | First author |
| Jiansong Zhang | School of Computer Science & Software Engineering, Shenzhen University, Shenzhen, China | 2453103003@mails.szu.edu.cn | Co-author |
| Pengfei Wei | School of Biological Science and Medical Engineering, Southeast University, Nanjing, China | 101014012@seu.edu.cn | Corresponding author |

Google Scholar profile: https://scholar.google.com/citations?hl=en&user=GPaRp8gAAAAJ

## Highlights

| Item | NanoSleepNet | NanoSleepNet-TCN | Causal NanoSleepNet-TCN |
|---|---:|---:|---:|
| Trainable parameters | 9.07K | 17.10K | 17.10K |
| FLOPs | 6.69M / epoch | 67.04M / 10-epoch window | 6.84M / online prediction |
| Input context | 1 epoch | 10 epochs | Current + previous 9 cached tokens |
| Intended use | Epoch-wise staging | Offline sequence-to-sequence staging | Online real-time staging |

The model uses five sleep stages: Wake, N1, N2, N3, and REM.

## Benchmark results

| Dataset | NanoSleepNet ACC | NanoSleepNet-TCN ACC | Causal ACC |
|---|---:|---:|---:|
| Sleep-EDF-20 | 82.69% | 84.25% | 84.12% |
| Sleep-EDF-78 | 79.49% | 81.89% | 82.06% |
| SHHS1 | 83.04% | 85.85% | 85.59% |

Temporal refinement improves accuracy by 1.56–2.81 percentage points over the epoch encoder in the reported experiments.

## On-device deployment

The manuscript evaluates FP32 and INT8 NCNN deployments on a Xiaomi 14 smartphone and Raspberry Pi Zero 2 W. INT8 NanoSleepNet-TCN inference reaches a mean latency of 1.45 ms on Xiaomi 14 and 3.93 ms on Raspberry Pi Zero 2 W in the reported benchmark setup.

The Android implementation described in the manuscript uses Kotlin/Jetpack Compose with a JNI-connected C++ NCNN backend. Raspberry Pi inference uses a native C++ benchmark program. The current public repository primarily contains PyTorch model definitions and the manuscript; deployment assets and full end-to-end benchmarking scripts should be added to the repository when ready for release.

## Architecture

The model has two stages. NanoSleepNet first extracts compact within-epoch morphology features from each 30-s, 100-Hz EEG epoch. A bottlenecked depthwise-separable TCN then refines a short sequence of 64-dimensional morphology tokens. In causal online mode, previous tokens are cached so that only the newly received EEG epoch must pass through the epoch encoder.

<img width="1413" height="779" alt="NanoSleepNet-TCN architecture" src="https://github.com/user-attachments/assets/4750bc90-17c7-4ee3-9b9b-d398d56ca772" />

## Repository contents

| File | Description |
|---|---|
| `Nanosleepnet_TCN.py` | NanoSleepNet-TCN sequence model and causal/non-causal TCN implementation |
| `NanoSeepNet.py` | NanoSleepNet epoch encoder implementation |
| `MSA_CNN.py` | MSA-CNN benchmark implementation/adaptation |
| `lwsleepnet.py` | LWSleepNet benchmark implementation/reimplementation |
| `lightsleepnet.py` | LightSleepNet benchmark implementation/reimplementation |
| `microsleepnet.py` | MicroSleepNet benchmark implementation/reimplementation |
| `sleepnet_lite.py` | SleepNet-Lite benchmark implementation/reimplementation |
| `efficientsleepnet.py` | EfficientSleepNet benchmark implementation/reimplementation |
| `tinysleepnet.py` | TinySleepNet benchmark implementation/adaptation |
| `deepsleepnet.py` | DeepSleepNet benchmark implementation/adaptation |
| `NanoSleepNet_TCN_completed.pdf` | Manuscript copy |

## Minimal usage

The main sequence implementation accepts a single epoch or a sequence of epochs. For 100-Hz EEG, a 30-s epoch contains 3000 samples.

```python
import torch
from Nanosleepnet_TCN import NanoSleepNetTCN

model = NanoSleepNetTCN(num_classes=5, causal=True)
model.eval()

# B=2, W=10, C=1, T=3000
x = torch.randn(2, 10, 1, 3000)

with torch.no_grad():
    logits = model(x)

print(logits.shape)  # (2, 10, 5)
```

For a minimal environment, the NanoSleepNet-TCN model itself requires PyTorch. Several benchmark files use additional packages such as NumPy and SciPy. A pinned `requirements.txt` or `environment.yml` should be added once the authors finalize the tested software versions.

## Datasets and evaluation protocol

The reported experiments use Sleep-EDF-20, Sleep-EDF-78, and SHHS1 under a unified five-stage setting. Sleep-EDF uses the Fpz–Cz EEG channel at 100 Hz. SHHS1 uses C4–A1 recorded at 125 Hz and is downsampled to 100 Hz before sequence construction.

The manuscript uses subject-level cross-validation with 20 folds for Sleep-EDF-20, 10 folds for Sleep-EDF-78, and 5 folds for SHHS1. Consecutive epochs are grouped into non-overlapping 10-epoch windows for the sequence experiments.

The datasets are not redistributed by this repository. Users are responsible for obtaining the data from the original providers and complying with the corresponding dataset access terms, licenses, and data-use agreements.

## Lightweight sleep-staging benchmark

This repository also collects implementations or reimplementations of lightweight sleep-staging baselines used under a unified experimental protocol, including MSA-CNN, ULW-SleepNet, LightSleepNet, MicroSleepNet, SleepNet-Lite, EfficientSleepNet, TinySleepNet, and DeepSleepNet.

Because benchmark files may be adapted from third-party projects or reimplemented from published architectures, their provenance and licensing must be tracked separately from the original NanoSleepNet-TCN code. See `THIRD_PARTY_NOTICES.md`.

## Funding

This work was supported in part by the National Natural Science Foundation of China under Grant T2394533, the Jiangsu Provincial Major Science and Technology Special Project under Grant BG2025040, the Jiangsu Provincial Advanced Technology Research and Development Program under Grant BF2025060, and the Start-up Research Fund of Southeast University under Grant RF1028625096.

## Citation

Until a final bibliographic record or DOI is available, please cite the manuscript and repository as follows:

```bibtex
@misc{liu2026nanosleepnettcn,
  title        = {NanoSleepNet-TCN: A Compact Temporal Model for On-Device Single-Channel EEG Sleep Staging},
  author       = {Liu, Guisong and Zhang, Jiansong and Wei, Pengfei},
  year         = {2026},
  note         = {Manuscript submitted to IEEE Transactions on Biomedical Engineering},
  url          = {https://github.com/guisongliu/Nano-SleepNet}
}
```

A machine-readable citation record is provided in `CITATION.cff`. Update the citation when the paper receives a DOI or final publication information.

## License

Original NanoSleepNet-TCN code and repository documentation are intended to be released under the Apache License 2.0. Benchmark implementations or third-party-derived files are not automatically relicensed under Apache-2.0 and remain subject to their upstream licenses, permissions, and attribution requirements. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

Before treating the entire repository as uniformly Apache-2.0 licensed, audit every benchmark file against its upstream source and preserve all required copyright notices and license texts.

## Contributing

Contributions that improve reproducibility, add verified baseline implementations, add deployment scripts, or fix documentation are welcome. Please read `CONTRIBUTING.md` before opening a pull request.

## Contact

For questions about the project, contact Guisong Liu at `230258331@seu.edu.cn`. For correspondence concerning the manuscript, the corresponding author is Pengfei Wei at `101014012@seu.edu.cn`.

## Disclaimer

This repository is intended for research and engineering use. It is not a medical device and is not intended to provide diagnosis, treatment, or clinical decision-making without appropriate validation and regulatory review.
