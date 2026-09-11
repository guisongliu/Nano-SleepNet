 NanoSleepNet-TCN: A Compact Temporal Model for On-Device Single-Channel EEG Sleep Staging

Author: Guisong Liu, Southeast University, Nanjing, China

My profile:https://scholar.google.com/citations?hl=en&user=GPaRp8gAAAAJ


Abstract:
On-device single-channel EEG sleep staging requires efficient modeling of both within-epoch morphology and inter-epoch temporal context. We propose NanoSleepNet_TCN, a compact sequence-to-sequence architecture that combines a 9.07K-parameter epoch encoder with lightweight token-level temporal refinement, resulting in only 17.10K parameters. Evaluated under subject-level protocols on Sleep-EDF-20, Sleep-EDF-78, and SHHS1, NanoSleepNet achieved accuracies of 84.25\%, 81.89\%, and 85.85\%, improving over the encoder alone by 1.56--2.81\% and outperforming all evaluated lightweight epoch-level baselines. A token-cached causal variant maintained comparable accuracy, with a mean absolute difference of 0.19\% from the non-causal model, while requiring 6.84M FLOPs per online prediction step. INT8 NCNN deployment achieved mean causal inference latencies of 1.45~ms on Android and 3.93~ms on Raspberry Pi Zero 2 W. These results indicate that the proposed method enables accurate and efficient real-time on-device sleep staging, thereby providing a practical foundation for sleep closed-loop modulation.


<img width="1413" height="779" alt="image" src="https://github.com/user-attachments/assets/4750bc90-17c7-4ee3-9b9b-d398d56ca772" />


This paper will be submitted to under reviewed.




###benchmarking Lightweight Sleep Staging Models###

We benchmark NanoSleepNet-TCN against a representative collection of lightweight sleep staging models under a unified training and evaluation protocol, enabling direct and fair comparison across different architectures. The benchmark covers lightweight non-sequential models, including MSA-CNN, ULW-SleepNet, LightSleepNet, MicroSleepNet, and SleepNet-Lite; lightweight temporal models, including EfficientSleepNet and TinySleepNet; as well as the classical CNN–sequence baseline DeepSleepNet.

To the best of our knowledge, this is the first systematically organized and fully open-access benchmark suite for lightweight sleep staging models. Beyond evaluating our proposed method, we hope this repository can serve as a common reference for the community, making lightweight sleep staging methods easier to reproduce, compare, and extend under consistent experimental settings.

We encourage future work to adopt transparent and standardized evaluation protocols so that comparisons are fair, reproducible, and meaningful, and we welcome the community to contribute additional lightweight baselines to continuously improve this benchmark.


Contact me via my email: 230258331@seu.edu.cn
