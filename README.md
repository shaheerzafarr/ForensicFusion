# ForensicFusion: Multi-Expert AI-Generated Image Forensic Detector

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaheerzafarr/ForensicFusion/blob/main/train_colab.ipynb)

**ForensicFusion** is an image-forensics framework designed to detect AI-generated, synthetic, and manipulated images across **33 generator architectures** by combining pretrained vision foundation models (**DINOv2**, **ConvNeXt V2**) with dedicated **pixel-level high-pass residual** and **2D frequency-domain (FFT)** forensic streams fused via cross-attention.

---

## ⚡ Quick Start on Google Colab (Recommended)

Click the badge above or open [`train_colab.ipynb`](./train_colab.ipynb) directly in Google Colab:
1. Set runtime to **T4 GPU** (*Runtime* > *Change runtime type* > *T4 GPU*).
2. Upload your `kaggle.json` API token to download the dataset in ~2 minutes.
3. Run the notebook to train with Automatic Mixed Precision (AMP) on NVIDIA Tensor Cores.

---

## 💻 Local Execution

### 1. Setup Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Settings
Adjust hyperparameters, batch size, and dataset paths in [`config.yaml`](./config.yaml).

### 3. Run Training
```powershell
python train.py
```

---

## 🧠 Architecture Overview
- **Spatial Semantic Stream**: DINOv2 Vision Transformer (`facebook/dinov2-small`)
- **Convolutional Stream**: ConvNeXt V2 (`facebook/convnextv2-tiny-1k-224`)
- **High-Pass Residual Stream**: Gaussian-filtered high-pass artifact CNN
- **Frequency Domain Stream**: 2D Fast Fourier Transform (FFT) log-magnitude CNN
- **Fusion**: Multi-head Cross-Attention token fusion with binary forensic and auxiliary generator heads