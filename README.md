# OmniPatch: A Universal Adversarial Patch for ViT-CNN Cross-Architecture Transfer in Semantic Segmentation
<p align="center">
  <img src="docs/assests/iclr-navbar-logo.svg" alt="ICLR" height="56"/>
  &nbsp;&nbsp;
  <img src="docs/assests/dsg-logo.png" alt="DSG IIITR" height="56"/>
  &nbsp;&nbsp;
  <img src="docs/assests/logo_iitr.svg" alt="IIT Roorkee" height="56"/>
</p>


<div align="center">

[![Workshop](https://img.shields.io/badge/ICLR_2026_Workshop-Principled_Design_for_Trustworthy_AI-blue.svg)](#)
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"/></a>
[![Paper](https://img.shields.io/badge/Omni-Patch-red.svg)](https://arxiv.org/abs/2603.20777)

</div>

---

## Abstract
Robust semantic segmentation is crucial for safe autonomous driving, yet deployed models remain vulnerable to black-box adversarial attacks when target weights are unknown. Most existing approaches either craft image-wide perturbations or optimize patches for a single architecture, which limits their practicality and transferability. We introduce \textbf{OmniPatch}, a training framework for learning a \emph{universal adversarial patch} that generalizes across images and both ViT and CNN architectures without requiring access to target model parameters.

---

## Method
![Method Pipeline](method_pipeline.jpg)

### Sensitive-Region Placement
Using a ViT surrogate, we compute class-wise predictive self-entropy on clean images and select the class with highest uncertainty. The patch is placed on high-uncertainty regions using entropy-biased sampling restricted to the top-$p\%$ of sensitive locations. This exploits the inductive bias gap between ViT global attention and CNN local feature extraction.

### Two-Stage Training
- **Stage 1 (ViT-only):** Optimize the patch to destabilize the ViT surrogate by targeting high-confidence predictions using a weighted cross-entropy.
- **Stage 2 (ViT+CNN Ensemble):** Extend training to a heterogeneous ensemble, mining high-transfer pixels (high Jensen-Shannon divergence) and weighting them relative to low-transfer regions to maximize cross-architecture transferability.

### Gradient Alignment
Standard ensemble training causes destructive gradient interference. We maximize cosine similarity between ViT and CNN gradients to homogenize update directions and prevent conflicting gradient flows.

### Auxiliary Losses & Physical Robustness
- **Attention Hijacking:** Force ViT to prioritize the patch over true labels in internal representations.
- **Boundary Disruption:** Induce fragmentation in segmentation boundaries.
- **Total Variation:** Noise control regularizer ensuring visual smoothness.
- **Physical Robustness:** Expectation-over-Transformation (EOT) modeling random scale, rotation, and translation.

---

## Results
Experiments are performed on the Cityscapes dataset using a 200×200 patch (1.9% area). OmniPatch achieves significant mIoU drops across diverse CNN and ViT architectures, demonstrating robust cross-architecture transferability.


| Model        | Clean mIoU | Random Patch | OmniPatch | mIoU Drop (%) |
| ------------ | ---------: | -----------: | --------: | -----------: |
| PIDNet-S     |    0.8695  |     0.8651   |  0.7299   |      15.96   |
| PIDNet-M     |    0.8681  |     0.8618   |  0.7393   |      14.84   |
| PIDNet-L     |    0.9035  |     0.8996   |  0.7530   |      16.65   |
| BiSeNetV1    |    0.7149  |     0.7057   |  0.6410   |      10.33   |
| BiSeNetV2    |    0.6907  |     0.6845   |  0.6036   |      12.61   |
| SegFormer    |    0.7434  |     0.7431   |  0.6777   |       8.83   |

---

## Repository Structure
```
.
├─ Experiments/                # Evaluation scripts
├─ configs/                    # YAML configurations
├─ dataset/                    # Data loaders
├─ metrics/                    # Evaluation metrics (mIoU, IOU)
├─ patch/                      # Patch priors and parameterization
├─ pretrained_models/          # Pretrained segmentation models
├─ trainer/
│  └─ trainer_TranSegPGD_AdvPatch.py  # Main OmniPatch trainer
├─ utils/                      # Utilities
└─ README.md
```

---

## Setup & Usage

### 1. Environment Setup
```bash
conda create -n omnipatch python=3.10 -y
conda activate omnipatch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Data Preparation
Download Cityscapes and set `CITYSCAPES_DIR=/path/to/cityscapes`. Expected structure:
```
CITYSCAPES_DIR/
  ├─ leftImg8bit/{train,val,test}/...
  └─ gtFine/{train,val,test}/...
```

### 3. Training & Evaluation
Training and evaluation notebooks are provided in the `kaggle/` folder:
- **`adversarial-patch-train.ipynb`**: Minimal patch baseline and sanity checks. Start here for a basic example.
- **`adv-patch-evaluation.ipynb`**: Full training pipeline and cross-architecture evaluation.

These notebooks contain the complete training loop using `trainer_TranSegPGD_AdvPatch.py` with all hyperparameters configured. Refer to them for reproducible runs and detailed parameter tuning.

---
## Acknowledgements

* **ICLR 2026 Workshop on Principled Design for Trustworthy AI** for accepting our paper.
* **Data Science Group (DSG), IIT Roorkee** for guidance and compute.
* Open‑source implementations of SegFormer, PIDNet, BiSeNet used for initialization/testing.

---
## License

This project is licensed under the terms of the **MIT License**.  
See the [LICENSE](LICENSE) file for full license text.

---
<!-- 
## Citation
If you find this work useful, please consider citing it as:
```bibtex
@inproceedings{aggarwal2026omnipatch,
  title={OmniPatch: A Universal Adversarial Patch for ViT-CNN Cross-Architecture Transfer in Semantic Segmentation},
  author={Aggarwal, Aarush and Tomar, Akshat and Goyal, Sargam and Tiwari, Amritanshu},
  booktitle={ICLR 2026 Workshop: Principled Design for Trustworthy AI},
  year={2026}
}
``` -->
