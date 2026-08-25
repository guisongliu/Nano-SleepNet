NanoSleepNet-TCN: Compact Epoch Encoding with Lightweight Token-Level Temporal Convolution for Single-Channel EEG Sleep Staging

Author: Guisong Liu, Southeast University, Nanjing, China

My profile:https://scholar.google.com/citations?hl=en&user=GPaRp8gAAAAJ


Abstract:
Single-channel EEG sleep staging is promising for home monitoring and wearable applications, but practical deployment requires accurate models with low computational cost. Existing methods mainly focus on lightweight epoch-encoder design, while efficient sequence-level temporal modeling remains insufficiently explored. We propose NanoSleepNet-TCN, a compact sequence-to-sequence architecture that jointly considers within-epoch morphology extraction and inter-epoch temporal refinement. The proposed 9.07K-parameter epoch encoder, NanoSleepNet, compresses each 30-s epoch into a 64-dimensional morphology token and achieves a favorable balance between performance and computational efficiency among the compared lightweight epoch encoders. These tokens are further refined by an 8.03K-parameter bottlenecked depthwise-separable TCN for low-cost token-level temporal refinement. Evaluated on Sleep-EDF-20, Sleep-EDF-78, and SHHS1, the complete 17.10K-parameter model achieved ACC values of 84.25\%, 81.89\%, and 85.85\%, respectively, with absolute ACC gains of 1.56--2.81\% over the epoch-only encoder. A causal variant retained comparable performance, supporting its potential for epoch-wise online inference. These results demonstrate that lightweight sleep staging benefits from both compact morphology-preserving epoch encoding and low-cost token-level temporal refinement.

<img width="1058" height="648" alt="image" src="https://github.com/user-attachments/assets/713f73d6-e8a8-4dd4-9a2d-b0145862def6" />



This paper will be submitted to under reviewed.




###benchmarking Lightweight Sleep Staging Models###

We benchmark NanoSleepNet-TCN against a representative collection of lightweight sleep staging models under a unified training and evaluation protocol, enabling direct and fair comparison across different architectures. The benchmark covers lightweight non-sequential models, including MSA-CNN, ULW-SleepNet, LightSleepNet, MicroSleepNet, and SleepNet-Lite; lightweight temporal models, including EfficientSleepNet and TinySleepNet; as well as the classical CNN–sequence baseline DeepSleepNet.

To the best of our knowledge, this is the first systematically organized and fully open-access benchmark suite for lightweight sleep staging models. Beyond evaluating our proposed method, we hope this repository can serve as a common reference for the community, making lightweight sleep staging methods easier to reproduce, compare, and extend under consistent experimental settings.

We encourage future work to adopt transparent and standardized evaluation protocols so that comparisons are fair, reproducible, and meaningful, and we welcome the community to contribute additional lightweight baselines to continuously improve this benchmark.


Contact me via my email: 230258331@seu.edu.cn
