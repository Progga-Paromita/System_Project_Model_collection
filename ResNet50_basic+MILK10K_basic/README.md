# ResNet-50 Fine-Tuning on MILK10K (ISIC Skin Lesion Dataset)

This repository contains a PyTorch implementation for downloading, indexing, and fine-tuning a pretrained **ResNet-50** on the official **MILK10K** skin lesion dataset in **Google Colab** without custom preprocessing or data augmentation.

---

## 📌 Project Overview
- **Base Model**: Pretrained `ResNet50` (`ResNet50_Weights.DEFAULT`)
- **Dataset**: Official ISIC MILK10K archive (`10.34970-648456`)
- **Task**: Multi-class skin lesion classification
- **Preprocessing**: None (Standard $224 \times 224$ resizing and ImageNet normalization only)
- **Architecture Changes**: None (Only replaced the final linear classification layer `model.fc`)

---

## 📁 Dataset & Pipeline Workflow

1. **Dataset Download & Extraction**:
   - Stream-downloads `milk10k.zip` from ISIC S3 archive.
   - Extracts images and metadata CSV to `/content/milk10k_dataset`.
2. **Metadata Indexing & Class Mapping**:
   - Automatically maps image IDs to disk file paths.
   - Encodes diagnostic labels into class indices dynamically.
3. **Data Splitting**:
   - Stratified **70% Train / 15% Validation / 15% Test** split.
4. **Fine-Tuning**:
   - Optimizer: `Adam` ($\text{lr} = 10^{-4}$)
   - Loss Function: `CrossEntropyLoss`
   - Batch Size: `32`
   - Epochs: `10`
5. **Model Checkpoint & Evaluation**:
   - Saves the best checkpoint based on validation accuracy to `/content/best_resnet50_milk10k.pth`.
   - Evaluates on the held-out test set (Accuracy, Macro-F1, Classification Report, Confusion Matrix).

---

## 🚀 Quickstart in Google Colab

### Step 1: Download & Extract Dataset
```python
import os, zipfile, requests

url = "https://isic-archive.s3.amazonaws.com/dois/10.34970-648456/milk10k.zip"
zip_path = "/content/milk10k.zip"
DATASET_DIR = "/content/milk10k_dataset"

# Download
response = requests.get(url, stream=True)
response.raise_for_status()
with open(zip_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            f.write(chunk)

# Extract
os.makedirs(DATASET_DIR, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(DATASET_DIR)
```

### Step 2: Run Fine-Tuning Script
Execute the training pipeline script. The notebook will automatically:
- Train the model across epochs.
- Save `best_resnet50_milk10k.pth`.
- Display final classification metrics and confusion matrix.

---

## ⚙️ Requirements
- Python 3.8+
- PyTorch & Torchvision
- pandas, numpy, scikit-learn
- matplotlib, seaborn, Pillow, requests
