# AGMS
AGMS: Adversarial Sample Generation-based Multi-Scale  Siamese Network for Hyperspectral Target Detection
# Run
PyTorch
1. Run get_bg.py clustering to get initial background samples.

TensorFlow
2. Generate target samples
2.1 Run PCA_bg.py to get the same number of background downscaled samples as the target prior.
2.2 Run GAN_target_AV.py to get the target generation samples.
3. Generate background samples
3.1 Run PCA.py to get the target sample dimensionality reduction.
3.2 Run GAN_bg.py to get the background generation samples.

PyTorch
4. Run train.py to train the multi-scale convolutional Siamese network using the generated samples and perform target detection.
# Citation
If you use code or datasets in this repository for your research, please cite our paper.
@ARTICLE{10731863,
  author={Luo, Fulin and Shi, Shanshan and Guo, Tan and Dong, Yanni and Zhang, Lefei and Du, Bo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={AGMS: Adversarial Sample Generation-Based Multiscale Siamese Network for Hyperspectral Target Detection}, 
  year={2024},
  volume={62},
  number={},
  pages={1-13},
  keywords={Training;Feature extraction;Object detection;Adaptation models;Generators;Detectors;Kernel;Interference;Minimization;Geoscience and remote sensing;Adversarial learning;deep learning;hyperspectral target detection (HTD);multiscale convolution;Siamese network},
  doi={10.1109/TGRS.2024.3484678}}
