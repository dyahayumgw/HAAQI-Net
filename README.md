# HAAQI-Net: A non-intrusive neural music quality assessment model for hearing aids

## Introduction
HAAQI-Net is a non-intrusive deep learning model for music quality assessment tailored to hearing aid users. In contrast to traditional methods like the Hearing Aid Audio Quality Index (HAAQI), HAAQI-Net utilizes a Bidirectional Long Short-Term Memory (BLSTM) with attention. It takes an assessed music sample and a hearing loss pattern as input, generating a predicted HAAQI score. The model employs the pre-trained Bidirectional Encoder representation from Audio Transformers (BEATs) for acoustic feature extraction. Comparing predicted scores with ground truth, HAAQI-Net achieves a Longitudinal Concordance Correlation (LCC) of 0.9257, Spearman’s Rank Correlation Coefficient (SRCC) of 0.9394, and Mean Squared Error (MSE) of 0.0080. Notably, this high performance comes with a substantial reduction in inference time: from 62.52 seconds (by HAAQI) to 2.71 seconds (by HAAQI-Net), serving as an efficient music quality assessment model for hearing aid users.

## Contributions
When designing HAAQI-Net, we focused on three key properties that achieve significant improvements over HAAQI:
1. **Non-intrusive:** HAAQI-Net predicts HAAQI scores based on corrupted signals and does not require clean references.
2. **Computationally Efficient:** HAAQI-Net is implemented using a simple neural network, enabling quality predictions to be computed in linear time.
3. **Differentiable:** Implemented as a neural network, HAAQI-Net can be incorporated into the loss function to train deep learning models for upstream tasks.

<p align="center">
  <img width="40%" src="https://github.com/dyahayumgw/HAAQI-Net/blob/main/pic/HAAQI-Net.png" alt="HAAQI-Net">
</p>

For more details and evaluation results, please check out our [HAAQI-Net Paper](https://arxiv.org/abs/2401.01145) and [dataset](https://t.ly/vLv29).
