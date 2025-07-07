# 🚀 ICCV 2025: PacGDC

[**PacGDC: Label-Efficient Generalizable Depth Completion with Projection Ambiguity and Consistency**](https://ieeexplore.ieee.org/document/10786388)

👨‍💻 **Haotian Wang, Aoran Xiao, Xiaoqin Zhang, Meng Yang, and Shijian Lu**

*International Conference on Computer Vision (ICCV), October 2025*

---

## 📝 Abstract

![examples](assets/teaser.png)

Generalizable depth completion enables the acquisition of dense metric depth maps for unseen environments, offering robust perception capabilities for various downstream tasks. However, training such models typically requires large-scale datasets with metric depth labels, which are often labor-intensive to collect.

**PacGDC** is a label-efficient technique that enhances data diversity with minimal annotation effort for generalizable depth completion. It leverages novel insights into inherent ambiguities and consistencies in object shapes and positions during 2D-to-3D projection, allowing the synthesis of numerous pseudo geometries for the same visual scene. This process greatly broadens data coverage by manipulating scene scales of the corresponding depth maps.

To leverage this property, we propose a new data synthesis pipeline that uses multiple depth foundation models as scale manipulators. These models robustly provide pseudo depth labels with varied scene scales in both local objects and global layouts, while ensuring projection consistency that contributes to generalization. To further diversify geometries, we incorporate interpolation and relocation strategies, as well as unlabeled images, extending the data coverage beyond the individual use of foundation models.

Extensive experiments show that PacGDC achieves remarkable generalizability across multiple benchmarks, excelling in diverse scene semantics/scales and depth sparsity/patterns under both zero-shot and few-shot settings.

---

## ⚙️ Requirements

- Python >= 3.9
- PyTorch >= 2.7

---

## 🏋️‍♂️ Training

### 1️⃣ Prepare Your Data

- Save your training datasets in `./Datasets/Data_Train`:

```
└── Data_Train
  ├── Labeled         # 📑 Labeled datasets
  │   ├── Dataset1
  │   │   ├── rgb
  │   │   │   ├── file1.png
  │   │   │   ├── file2.png
  │   │   │   └── ...
  │   │   ├── depth
  │   │   │   ├── file1.png
  │   │   │   ├── file2.png
  │   │   │   └── ...
  │   │   ├── DA      # 🟦 DepthAnything Results
  │   │   │   ├── file1.png
  │   │   │   ├── file2.png
  │   │   │   └── ...
  │   │   └── DepthPro # 🟩 DepthPro Results
  │   │       ├── file1.png
  │   │       ├── file2.png
  │   │       └── ...
  │   └── Dataset2
  │       └── ...
  └── UnLabeled       # 🕶️ Unlabeled datasets
    ├── Dataset1
    │   ├── rgb
    │   │   └── ...
    │   ├── DA
    │   │   └── ...
    │   ├── DepthPro
    │   │   └── ...
    └── Dataset2
      └── ...
```

> ⚠️ **Note:**  
> - `depth` should be stored in 16-bit format, normalized by `depth(m)/max_depth(m)*65535`.  
> - `max_depth=30(m)` for indoor datasets, `max_depth=150(m)` for outdoor datasets.

- Save your hole datasets in `./Datasets/Data_Hole`:

```
└── Hole_Datasets
  ├── Dataset1
  │   ├── file1.png
  │   ├── file2.png
  │   └── ...
  └── Dataset2
    ├── file1.png
    ├── file2.png
    └── ...
```

> ⚠️ **Note:**  
> - Hole maps should be stored in Uint8 format.  
> - `valid pixels = 255`, `invalid pixels = 0`.  
> - Example: [hole collected from HRWSI](https://drive.google.com/file/d/1iKJEWgd36ebEVbG-01_gDipYuCCs7ZQZ/view?usp=drive_link)

---

### 2️⃣ Start Training

- Run `train.py`:

```bash
# model_type: ["T", "S", "B", "L"] referring to SPNet
python train.py --model_type="L" --foundation_models="DA_DepthPro"
```

- The trained model will be saved in `./logs/models`

---

## 🧪 Testing
1. 📥 Download and save the `Zero-Shot Pretrained Model` to `./Pretrained`

| Pretrained Model                                                                                    | Model Type    | Drop rate |
| --------------------------------------------------------------------------------------------------- |:-------:|:-------:|
| [Zero-shot](https://drive.google.com/file/d/1QlZhWOFkF-Penz1fYz6gyE3AxzrFdT6j/view?usp=drive_link)    | SPNet-Large      | 0.5  |
| [KITTI Finetuned (Online Leaderboard)](https://drive.google.com/file/d/1_9NnvnfeCcgAmIGnAXB8VwPlj1kz8hFD/view?usp=drive_link)   | SPNet-Large     | 0.8  |

2. 📥 Download and unzip [Test Dataset (Ibims)](https://drive.google.com/file/d/10tME1cuV0PVxrFLauTlv5SdQbZLUfdGy/view?usp=drive_link) to `./Datasets/Data_Test`
3. ▶️ Run `test.py`:

```bash
# 1. Normalize depth values to [0,1] by "depth(m)/max_depth(m)*65535"
# (The provided "Test Dataset (Ibims)" is already normalized with max_depth=30 (Indoor))

# 2. Run test.py
python test.py --ckpt_path="Pretrained/L_DA_DepthPro.pth" --max_depth=30
```

---

## 📚 Citation

If you find our work useful, please cite:

```bibtex
@ARTICLE{10786388,
  author={Wang, Haotian and Yang, Meng and Zheng, Xinhu and Hua, Gang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Scale Propagation Network for Generalizable Depth Completion}, 
  year={2025},
  volume={47},
  number={3},
  pages={1908-1922},
  doi={10.1109/TPAMI.2024.3513440}
}
```
