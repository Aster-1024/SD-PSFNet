# SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining

<p align="center">
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37957"><b>Paper</b></a>
  &nbsp;|&nbsp;
  <a href="https://doi.org/10.1609/aaai.v40i12.37957"><b>DOI</b></a>
</p>
<p align="center">
  <b>🎆 AAAI 2026</b>
</p>


<p align="center">
  Jiayu Wang<sup>1*</sup>, Haoyu Bian<sup>2,3*</sup>, Haoran Sun<sup>1</sup>, Shaoning Zeng<sup>2,3†</sup>
</p>

<p align="center">
  <sup>1</sup>School of Information and Software Engineering, University of Electronic Science and Technology of China, Chengdu, China<br>
  <sup>2</sup>Yangtze Delta Region Institute (Huzhou), University of Electronic Science and Technology of China, Huzhou, China<br>
  <sup>3</sup>School of Computer Science and Engineering, University of Electronic Science and Technology of China, Chengdu, China
</p>
**✨ Highlights**

SD-PSFNet introduces a physics-aware deraining framework that models rain streak degradation with learned Point Spread Function (PSF) mechanisms. The network progressively restores rain-free images through cascaded stages and adaptive gated feature fusion.

---

## News

- 🔥 **2026-03-14**: SD-PSFNet was published in the Proceedings of the AAAI Conference on Artificial Intelligence.

---

## Overview

<p align="center">
  <img src="image/main.svg" width="95%" alt="Overall architecture of SD-PSFNet">
</p>

SD-PSFNet is designed for challenging rain removal scenarios. It contains three key ideas:

1. **Sequential restoration**: the model removes rain progressively from coarse restoration to fine detail recovery.
2. **Dynamic PSF modeling**: learned PSF components simulate rain streak optics and help separate rain degradation from background content.
3. **Adaptive gated fusion**: cross-stage features are selectively transferred to improve restoration consistency and preserve image details.

---

## Stage-wise Restoration

<p align="center">
  <img src="image/stage1.svg" width="95%" alt="Stage-wise restoration process">
</p>

The sequential design enables multiple dynamic evaluations and refinements of the degradation process. Earlier stages focus on coarse rain removal, while later stages refine textures, structures, and clean image details.

---

## Visual Results

<p align="center">
  <img src="image/result1.svg" width="95%" alt="Qualitative deraining results">
</p>

---

## Repository Structure

```text
SD-PSFNet/
├── image/
│   ├── main.svg
│   ├── result1.svg
│   └── stage1.svg
├── models/
├── weights/
│   ├── Rain100H-sota.pth
│   ├── Rain100L-sota.pth
│   ├── Realrain-1k-H-sota.pth
│   └── Realrain-1k-L-sota.pth
├── Data.py
├── Loss.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

---

## Requirements

Create a new conda environment and install the required packages:

```bash
conda create -n sdpsfnet python=3.8 -y
conda activate sdpsfnet

pip install -r requirements.txt
```

If PyTorch is not included in `requirements.txt`, install the version that matches your CUDA environment from the official PyTorch installation page.

---

## Dataset Preparation

Please prepare the datasets used for image deraining, such as **Rain100H**, **Rain100L**, **RealRain-1k-H**, and **RealRain-1k-L**.

A recommended dataset structure is:

```text
datasets/
├── Rain100H/
│   ├── train/
│   │   ├── rainy/
│   │   └── gt/
│   └── test/
│       ├── rainy/
│       └── gt/
├── Rain100L/
│   ├── train/
│   │   ├── rainy/
│   │   └── gt/
│   └── test/
│       ├── rainy/
│       └── gt/
├── RealRain-1k-H/
│   ├── train/
│   │   ├── rainy/
│   │   └── gt/
│   └── test/
│       ├── rainy/
│       └── gt/
└── RealRain-1k-L/
    ├── train/
    │   ├── rainy/
    │   └── gt/
    └── test/
        ├── rainy/
        └── gt/
```

Before training or testing, please check the dataset path settings in `Data.py` and modify them according to your local directory.

---

## Pre-trained Models

Place the pre-trained checkpoints in the `weights/` directory.

| Dataset | Checkpoint |
| --- | --- |
| Rain100H | `weights/Rain100H-sota.pth` |
| Rain100L | `weights/Rain100L-sota.pth` |
| RealRain-1k-H | `weights/Realrain-1k-H-sota.pth` |
| RealRain-1k-L | `weights/Realrain-1k-L-sota.pth` |

---

## Testing

To evaluate SD-PSFNet with the provided pre-trained checkpoints, first confirm the dataset path and checkpoint path in `test.py`, then run:

```bash
python test.py
```

The restored images will be saved to the output directory defined in `test.py`.

If your implementation supports command-line arguments, you can adapt the command as follows:

```bash
python test.py \
  --dataset Rain100H \
  --weights weights/Rain100H-sota.pth \
  --input_dir datasets/Rain100H/test/rainy \
  --output_dir results/Rain100H
```

---

## Training

To train SD-PSFNet from scratch, first prepare the training data and check the training configuration in `train.py` and `Data.py`.

Then run:

```bash
python train.py
```

Training logs and checkpoints will be saved according to the path settings in `train.py`.

---

## Results

The paper reports state-of-the-art PSNR/SSIM performance on several deraining benchmarks.

| Dataset | PSNR | SSIM |
| --- | ---: | ---: |
| Rain100H | 33.12 dB | 0.9371 |
| RealRain-1k-L | 42.28 dB | 0.9872 |
| RealRain-1k-H | 41.08 dB | 0.9838 |

To reproduce the reported results, please use the corresponding pre-trained checkpoint and the official benchmark test split.

---

## Citation

If you find this project useful for your research, please cite:

```bibtex
@inproceedings{wang2026sd,
  title={SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining},
  author={Wang, Jiayu and Bian, Haoyu and Sun, Haoran and Zeng, Shaoning},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={12},
  pages={9921--9929},
  year={2026}
}
```

---

## Acknowledgements

This work was supported by the Zhejiang Provincial Natural Science Foundation of China (Grant NO. LY23F020025), the Science and Technology Program of Huzhou (Grant NO. 2024GZ09), the Zhejiang Province Leading Geese Plan (Grant NO. 2025C02025), and the National Natural ScienceFoundation of China (Grant NO. 62576292).

---

## Contact

For questions about the paper or implementation, please open an issue or contact the authors.
        Primary contact: Jiayu Wang([2024090902024@std.uestc.edu.cn](mailto:2024090902024@std.uestc.edu.cn))
