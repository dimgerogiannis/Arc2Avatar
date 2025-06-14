# Arc2Avatar: Generating Expressive 3‑D Avatars from a Single Image via ID Guidance

<p align="center"><img src="./assets/method_figure_placeholder.png" width="760" alt="Method overview placeholder"></p>

<font color="#1F618D"><strong>Dimitrios Gerogiannis</strong></font> · <font color="#1F618D"><strong>Foivos Paraperas Papantoniou</strong></font> · <font color="#1F618D"><strong>Rolandos Alexandros Potamias</strong></font> · <font color="#1F618D"><strong>Alexandros Lattas</strong></font> · <font color="#1F618D"><strong>Stefanos Zafeiriou</strong></font>  
<font color="#117A65"><em>Imperial College London, UK</em></font>  
<font color="#2980B9">{d.gerogiannis22, f.paraperas, r.potamias, a.lattas, s.zafeiriou}@imperial.ac.uk</font>

---

## 🗞️ News
- **14 June 2025** – *Initial public release*: full training/inference code **and pretrained models** are now available.  
- **Coming soon** – Expression‑control fine‑tuning code and weights will be added in a follow‑up commit.

---

## ✨ Introduction
- **Text‐conditioned distillation is too abstract** for identity‑preserving face reconstruction; we guide SDS with dense *ArcFace* embeddings instead.
- **First to couple a human‑face foundation model with SDS**; prior art (*ID‑to‑3D*) used ArcFace vectors but not a full frozen face network.
- **Strategic, low‑guidance SDS** + a strong face prior tame oversaturation, yielding avatars with natural colour fidelity.

---

## ⚙️ Installation
```bash
# 1 · Conda shell (GPU build, CUDA 11.8)
conda create -n Arc2Avatar python=3.9 cudatoolkit=11.8 -y
conda activate Arc2Avatar

# 2 · Python stack
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# 3 · Local CUDA extensions
env PIP_NO_BUILD_ISOLATION=1 python -m pip install   --config-settings editable_mode=compat   -e submodules/diff-gaussian-rasterization   -e submodules/simple-knn
```
*CPU‑only?* Remove `cudatoolkit=11.8` and install the `+cpu` Torch wheels.

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
```bibtex
@inproceedings{gerogiannis2025arc2avatar,
  title     = {Arc2Avatar: Generating Expressive 3-\!D Avatars from a Single Image via ID Guidance},
  author    = {Gerogiannis, Dimitrios and Paraperas Papantoniou, Foivos and Potamias, Rolandos Alexandros and Lattas, Alexandros and Zafeiriou, Stefanos},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```

---
