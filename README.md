# AGMS
AGMS: Adversarial Sample Generation-based Multi-Scale  Siamese Network for Hyperspectral Target Detection
# Overview
![Uploading 22.png…]()

# Run
(1 and 4 Using the PyTorch environment)
(2 and 3 Using the TensorFlow environment)
1. Run get_bg.py clustering to get initial background samples.
2. Generate target samples
2.1 Run PCA_bg.py to get the same number of background downscaled samples as the target prior.
2.2 Run GAN_target_AV.py to get the target generation samples.
3. Generate background samples
3.1 Run PCA.py to get the target sample dimensionality reduction.
3.2 Run GAN_bg.py to get the background generation samples.
4. Run train.py to train the multi-scale convolutional Siamese network using the generated samples and perform target detection.
   
# Citation
If you use code or datasets in this repository for your research, please cite our paper.

F. Luo, S. Shi, T. Guo, Y. Dong, L. Zhang and B. Du, "AGMS: Adversarial Sample Generation-Based Multiscale Siamese Network for Hyperspectral Target Detection," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-13, 2024, Art no. 5536713, doi: 10.1109/TGRS.2024.3484678. 
