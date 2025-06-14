Arc2Avatar: Generating Expressive 3‑D Avatars from a Single Image via ID Guidance

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
  <a href="https://arxiv.org/abs/2406.12345" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Paper-arXiv-d9534f?style=for-the-badge&logo=arxiv" alt="arXiv Paper"/>
  </a>
</p>



⸻

🗞️ News
	•	14 June 2025 – Initial public release: full training/inference code and pretrained models are now available.
	•	Coming soon – Expression‑control fine‑tuning code and weights will be added in a follow‑up commit.

⸻

✨ Introduction
	•	Text‐conditioned distillation is too abstract for identity‑preserving face reconstruction; we guide SDS with dense ArcFace embeddings instead.
	•	First to couple a human‑face foundation model with SDS; prior art (ID‑to‑3D) used ArcFace vectors but not a full frozen face network.
	•	Strategic, low‑guidance SDS + a strong face prior tame oversaturation, yielding avatars with natural colour fidelity.

⸻

⚙️ Installation

# 1 · Conda shell (GPU build, CUDA 11.8)
conda create -n Arc2Avatar python=3.9 cudatoolkit=11.8 -y
conda activate Arc2Avatar

# 2 · Python stack
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# 3 · Local CUDA extensions
env PIP_NO_BUILD_ISOLATION=1 python -m pip install \
  --config-settings editable_mode=compat \
  -e submodules/diff-gaussian-rasterization \
  -e submodules/simple-knn

CPU‑only? Remove cudatoolkit=11.8 and install the +cpu Torch wheels.

⸻

🚀 Usage

Quick demo

python demo.py --img ./assets/face.jpg --out out/ --exp "happy"
open out/index.html

Train your own avatar

python train.py \
  --img ./data/my_subject.jpg \
  --epochs 600 --batch 4 \
  --lora_rank 16 --guidance_scale 2.0 \
  --out runs/my_subject


⸻

📚 Citation

@inproceedings{gerogiannis2025arc2avatar,
  title     = {Arc2Avatar: Generating Expressive 3-\!D Avatars from a Single Image via ID Guidance},
  author    = {Gerogiannis, Dimitrios and Paraperas Papantoniou, Foivos and Potamias, Rolandos Alexandros and Lattas, Alexandros and Zafeiriou, Stefanos},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}


⸻
