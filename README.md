NanoSleepNet-TCN: Compact Epoch Encoding with Lightweight Token-Level Temporal Convolution for Single-Channel EEG Sleep Staging

Author: Guisong Liu, Southeast University, Nanjing, China

Abstract:
Single-channel EEG sleep staging is promising for home monitoring and wearable applications, but practical deployment requires accurate models with low computational cost. Existing methods mainly focus on lightweight epoch-encoder design, while efficient sequence-level temporal modeling remains insufficiently explored. We propose NanoSleepNet-TCN, a compact sequence-to-sequence architecture that jointly considers within-epoch morphology extraction and inter-epoch temporal refinement. The proposed 9.07K-parameter epoch encoder, NanoSleepNet, compresses each 30-s epoch into a 64-dimensional morphology token and achieves a favorable balance between performance and computational efficiency among the compared lightweight epoch encoders. These tokens are further refined by an 8.03K-parameter bottlenecked depthwise-separable TCN for low-cost token-level temporal refinement. Evaluated on Sleep-EDF-20, Sleep-EDF-78, and SHHS1, the complete 17.10K-parameter model achieved ACC values of 84.25\%, 81.89\%, and 85.85\%, respectively, with absolute ACC gains of 1.56--2.81\% over the epoch-only encoder. A causal variant retained comparable performance, supporting its potential for epoch-wise online inference. These results demonstrate that lightweight sleep staging benefits from both compact morphology-preserving epoch encoding and low-cost token-level temporal refinement.


This paper will be submitted to under reviewed.
