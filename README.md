 NanoSleepNet-TCN: A Compact Temporal Model for On-Device Single-Channel EEG Sleep Staging

Author: Guisong Liu, Pengfei Wei Southeast University, Nanjing, China

Jiansong Zhang  Shenzhen University, Shenzhen, China

My profile:https://scholar.google.com/citations?hl=en&user=GPaRp8gAAAAJ


Abstract:
Precise sleep modulation relies on real-time sleep staging directly on edge devices. On-device EEG sleep staging requires accurate temporal modeling under strict hardware constraints. We propose NanoSleepNet-TCN, which compresses each EEG epoch into a compact morphology token and performs lightweight temporal refinement in the token space. The model contains only 17.10K parameters, including a 9.07K-parameter epoch encoder, and supports token-cached causal inference. On Sleep-EDF-20, Sleep-EDF-78, and SHHS1 datasets, NanoSleepNet-TCN achieved accuracies of 84.25\%, 81.89\%, and 85.85\%, respectively, with temporal refinement improving accuracy by 1.56--2.81\% over the encoder alone. The causal model requires only 6.84M FLOPs per online prediction and maintains comparable accuracy to the non-causal model. INT8 deployment achieves mean inference latencies of 1.45~ms on a Xiaomi 14 smartphone and 3.93~ms on a Raspberry Pi Zero 2 W. These results demonstrate accurate and efficient real-time temporal sleep staging on resource-constrained devices. This work can directly support the implementation of applications such as sleep closed-loop modulation. The source code is available at https://github.com/guisongliu/Nano-SleepNet.

<img width="1413" height="779" alt="image" src="https://github.com/user-attachments/assets/4750bc90-17c7-4ee3-9b9b-d398d56ca772" />


This paper has been submitted to IEEE Transactions on Biomedical Engineering.




###benchmarking Lightweight Sleep Staging Models###

We benchmark NanoSleepNet-TCN against a representative collection of lightweight sleep staging models under a unified training and evaluation protocol, enabling direct and fair comparison across different architectures. The benchmark covers lightweight non-sequential models, including MSA-CNN, ULW-SleepNet, LightSleepNet, MicroSleepNet, and SleepNet-Lite; lightweight temporal models, including EfficientSleepNet and TinySleepNet; as well as the classical CNN–sequence baseline DeepSleepNet.

To the best of our knowledge, this is the first systematically organized and fully open-access benchmark suite for lightweight sleep staging models. Beyond evaluating our proposed method, we hope this repository can serve as a common reference for the community, making lightweight sleep staging methods easier to reproduce, compare, and extend under consistent experimental settings.

We encourage future work to adopt transparent and standardized evaluation protocols so that comparisons are fair, reproducible, and meaningful, and we welcome the community to contribute additional lightweight baselines to continuously improve this benchmark.


Contact me via my email: 230258331@seu.edu.cn
