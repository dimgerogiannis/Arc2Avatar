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
## ⚙️ Installation

```bash
# 1 · Clone submodules with recursive flags
cd submodules
git clone --recursive https://github.com/YixunLiang/diff-gaussian-rasterization.git
git clone --recursive https://github.com/YixunLiang/simple-knn.git
cd ..

# 2 · Create Conda environment (GPU setup, CUDA 11.8)
conda create -n arc2avatar python=3.9.16 cudatoolkit=11.8 -y
conda activate arc2avatar

# 3 · Install Python dependencies
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# 4 · Build local CUDA extensions
python -m pip install submodules/diff-gaussian-rasterization/
python -m pip install submodules/simple-knn/

# 5 · Download Arc2Face models
python download_models.py

---

## 🚀 Usage
### Quick demo
```bash
python demo.py --img ./assets/face.jpg --out out/ --exp "happy"
open out/index.html
```

### Train your own avatar
```bash
python train.py   --img ./data/my_subject.jpg   --epochs 600 --batch 4   --lora_rank 16 --guidance_scale 2.0   --out runs/my_subject
```

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
