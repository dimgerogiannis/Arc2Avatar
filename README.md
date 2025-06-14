<div align="center">
  <h1>Arc2Avatar: Generating Expressive 3‑D Avatars from a Single Image via ID Guidance</h1>
</div>

<p align="center"><img src="./assets/teaser.png" width="1000" alt="Method overview placeholder"></p>

<p align="center">
  <a href="https://dimgerogiannis.github.io/" style="color:#1a73e8;">Dimitrios Gerogiannis</a>,
  <a href="https://foivospar.github.io" style="color:#1a73e8;">Foivos Paraperas Papantoniou</a>,
  <a href="https://rolpotamias.github.io" style="color:#1a73e8;">Rolandos Alexandros Potamias</a>,
  <a href="https://alexlattas.com" style="color:#1a73e8;">Alexandros Lattas</a>,
  <a href="https://profiles.imperial.ac.uk/s.zafeiriou" style="color:#1a73e8;">Stefanos Zafeiriou</a><br>
  <span style="color:#1a73e8;">Imperial College London, UK</span>
</p>

<p align="center">
  <a href="https://arc2avatar.github.io" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Project-Page-1a73e8?style=for-the-badge&logo=github" alt="Project Page"/>
  </a>
  <a href="https://arxiv.org/abs/2501.05379" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Paper-arXiv-d9534f?style=for-the-badge&logo=arxiv" alt="arXiv Paper"/>
  </a>
</p>

---

## ✨ Introduction

**Arc2Avatar is an SDS-based method that generates a complete 3D head from a single image**, delivering:

- 🔥 **avatars of unprecedented realism, detail, and natural color fidelity**, while avoiding the common color issues of SDS.  
- 🔥 **first approach to leverage a human face foundation model** as guidance.  
- 🔥 **full 3DMM integration**, enabling expression control and refinements within the same framework.  
- 🔥 **state-of-the-art identity preservation and superior overall quality**, supported by both quantitative and qualitative results.  

---

## 🗞️ News
- **14 June 2025** – *Initial public release*: full training/inference code **and pretrained models** are now available.  
- **Coming soon** – Expression‑control fine‑tuning code and weights will be added in a follow‑up commit.

---

## ⚙️ Installation

**Step 1 – Clone the CUDA extension submodules**

```bash
cd submodules
git clone --recursive https://github.com/YixunLiang/diff-gaussian-rasterization.git
git clone --recursive https://github.com/YixunLiang/simple-knn.git
cd ..
```

**Step 2 – Create and activate the conda environment (CUDA 11.8 and Python 3.9.16, tested on NVIDIA RTX 4090)**

```bash
conda create -n arc2avatar python=3.9.16 cudatoolkit=11.8 
conda activate arc2avatar
```

**Step 3 – Install the dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 – Build and install local CUDA extensions**

```bash
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
```

**Step 5 – Download required models for Arc2Face using the following script**

```bash
python download_models.py
```

---

## 🚀 Usage

To train your own 3D avatars 🎭, follow these steps:

1. **Create a subject directory**  
   Make a new folder for your subject (e.g. `subject_id`) and place a single image of the individual inside it.  
   `subject_id` refers to a **path** to this subject-specific directory.

2. **Run the training script**

```bash
python train.py --opt ./configs/config.yaml --subject subject_id --batch_size 4
```

3. **Track optimization progress**  
   During training, a subfolder named `splat/` is automatically created inside `subject_id/`.  
   This directory contains the evolving 3D Gaussian avatar, allowing you to visually monitor the SDS process from start to finish.

---

## 📚 Citation

If you find Arc2Avatar useful for your research, please consider citing our paper:

```bibtex
@article{gerogiannis2025arc2avatar,
  title={Arc2Avatar: Generating Expressive 3D Avatars from a Single Image via ID Guidance},
  author={Gerogiannis, Dimitrios and Papantoniou, Foivos Paraperas and Potamias, Rolandos Alexandros and Lattas, Alexandros and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:2501.05379},
  year={2025}
}
```

---
