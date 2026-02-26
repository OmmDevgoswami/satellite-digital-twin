# Satellite-Digital-Twin: Real-Time Environmental Intelligence Platform (v3.9.2)

**Satellite-Digital-Twin** is a production-grade satellite telemetry platform for environmental intelligence. AI-powered waste-dump detection with a multi-model ensemble, geospatial analytics and ESG impact quantification. It combines an EfficientNet-B4 classifier with a 3-model segmentation ensemble, streams incoming imagery via the Pathway framework, and produces GIS-mapped ESG impact metrics. The codebase includes a Glassmorphic Streamlit UI, multi-language support, and a Dockerfile/`vercel.json` for easy deployment.

> Built as the Hack For Green Bharat submission; extendable to any telemetry use-case.

---

## 🏆 Competition Edge: The Elite Digital Twin (v3.9.2)
*   **Pathway Live AI Engine**: Real-time incremental processing of satellite telemetry using the high-performance Pathway streaming framework (Linux environments).
*   **Dual-Task AI Integration**: Specialized binary classification + pixel-level segmentation (FPN, UNet++, DeepLabV3+) capable of identifying illegal dumps in complex natural terrains.
*   **Sustainability Intelligence**: Automated estimation of waste tonnage (@150 T/Ha), CO₂ potential (@1.2 kg/T), and municipal remedial costs (₹2,200/T).
*   **Elite Glassmorphism UI**: A professional, enterprise-grade Streamlit interface with full theme-aware (Light/Dark) reactivity.

---

## 🌟 Core Technical Features

### 📊 Advanced Deep Learning Suite
- **Classification Backbone**: **EfficientNet-B4** for binary dump/no-dump classification with adaptive pooling and dropout regularization.
- **Multi-Architecture Segmentation**: Ensemble of **FPN**, **UNet++**, and **DeepLabV3+** encoders for pixel-level dump boundary detection.
- **Performance**: 
  - **FPN**: IoU = 0.207, Dice = 0.314
  - **UNet++**: IoU = 0.201, Dice = 0.311
  - **DeepLabV3+**: IoU = 0.193, Dice = 0.297
- **Test-Time Augmentation (TTA)**: Predictions stabilized across 5 geometric views for enhanced robustness.
- **Robust Loss Functions**: Focal and Tversky loss implementations to handle class imbalance in satellite imagery.

### 🗺️ Geo-Spatial Intelligence
- **Folium Integration**: High-resolution satellite tiles with marker clustering and heatmap overlays for national monitoring.
- **EXIF Auto-Extraction**: Automated GPS coordinate retrieval from uploaded image metadata for precise mapping.
- **Regional Zone Analysis**: Comparative risk assessments across territorial regions (North, South, East, West).

### 🌿 Environmental Impact Engine
- **Severity Index (0-100)**: Proprietary weighted score combining:
  - Classifier confidence (40%)
  - Spatial coverage percentage (40%)
  - Estimated physical area in hectares (20%)
- **Environmental Metrics**: Real-time calculation of:
  - **Waste Tonnage**: Based on 150 T/Ha density assumption
  - **CO₂ Equivalent Emissions**: ~1.2 kg CO₂e per Tonne
  - **Cleanup Cost Estimates**: ₹2,200 per Tonne (Indian benchmark)
- **PDF Report Generation**: GIS-encoded single-page analysis reports for municipal archival.

---

## 📁 Repository Architecture
```
d:/satellite-dump-detection-main/
├── app/                      # Streamlit Web Application
│   ├── app.py                # Main Glassmorphic UI Portal
│   └── translations.py       # Multi-language support (EN/HI/TE)
├── src/
│   ├── data/                 # Dataset loaders & preprocessing
│   │   ├── dataset.py        # AerialWaste dataloader
│   │   └── transforms.py     # Albumentations-based augmentations
│   ├── models/               # Neural network architectures
│   │   ├── classifier.py     # Binary classification (ResNet34/50, EfficientNet-B4)
│   │   └── segmentation_model.py  # Segmentation (UNet, UNet++, FPN, DeepLabV3+)
│   ├── training/             # Training pipelines
│   │   ├── train_classifier.py
│   │   ├── train_segmentation.py
│   │   ├── train_efficientnet.py
│   │   ├── train_advanced_seg.py
│   │   ├── trainer.py        # Generic trainer class
│   │   ├── evaluate.py       # Evaluation metrics
│   │   └── metrics.py        # Custom metric definitions
│   ├── streaming/            # Real-time processing
│   │   ├── pathway_pipeline.py   # Pathway framework integration (Linux)
│   │   └── folder_stream.py      # Local folder streaming (Windows/Mac)
│   └── utils/                # Utilities
│       ├── config.py         # Centralized configuration
│       ├── severity.py       # Severity scoring & impact estimation
│       └── visualize.py      # Visualization helpers
├── data/
│   ├── raw/
│   │   └── AerialWaste/      # Official dataset folder
│   │       ├── training.json
│   │       ├── testing.json
│   │       └── images/       # Image subdirectories
│   └── stream_incoming/      # Live stream input directory
├── notebooks/                # Data exploration & debugging
│   ├── 01_eda.py
│   ├── 02_verify_dataloader.py
│   └── debug_dataset.py
├── outputs/
│   ├── checkpoints/          # Trained model weights (.pt files)
│   │   ├── best_classifier.pt
│   │   ├── best_efficientnet.pt
│   │   ├── best_fpn.pt
│   │   ├── best_unetplusplus.pt
│   │   └── evaluation/       # Performance metrics & results
│   └── pathway/              # Streaming output (live_events.csv)
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment specification
├── README.md                 # This file
├── FEATURE_GUIDE.md          # User guide & feature documentation
└── IMPACT_SUMMARY.md         # Hackathon pitch assets
```

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Create conda environment
conda create -n dump_detect python=3.10 -y
conda activate dump_detect

# Install PyTorch (CUDA 11.8 recommended for GPU acceleration)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
- Extract the **AerialWaste** dataset into `data/raw/AerialWaste/`
- Ensure the folder structure contains:
  - `training.json` (metadata with category labels)
  - `testing.json` (test set with polygon masks)
  - `images/images0/` (satellite images)

### 3. Launch the Application
```bash
conda activate dump_detect
streamlit run app/app.py
```
The application will open at `http://localhost:8501` with the Glassmorphic UI.

### 4. Training Models (Optional)
```bash
# Train classifier
python src/training/train_efficientnet.py

# Train segmentation models
python src/training/train_advanced_seg.py
```

---

## 🎯 Key Features

### Classification Pipeline
- **Input**: RGB satellite image (256×256)
- **Model**: EfficientNet-B4 backbone with binary output
- **Output**: Dump probability [0, 1]

### Segmentation Pipeline
- **Input**: RGB satellite image (256×256)
- **Models**: Multi-architecture ensemble (FPN, UNet++, DeepLabV3+)
- **Output**: Binary mask (0 = no dump, 1 = dump) with spatial coverage %

### Severity Scoring
- **Formula**: `Score = 0.4×prob + 0.4×(coverage/100) + 0.2×min(area/10, 1)`
- **Levels**: LOW (🟢 0-25) | MEDIUM (🟡 25-50) | HIGH (🟠 50-75) | CRITICAL (🔴 75-100)

### Environmental Impact Estimation
- Calculates estimated waste tonnage from spatial coverage × area × density assumption
- Estimates CO₂ equivalent using waste-to-emission conversion coefficients
- Projects cleanup costs using ₹2,200/Tonne municipal benchmark

---

## 📊 Technical Specifications

### Dataset: AerialWaste
- **Total Images**: Multiple subsets (images0, images1, images2, etc.)
- **Image Size**: 256×256 pixels (auto-resized in pipeline)
- **Labels**: Binary (dump/no-dump) + polygon segmentation masks (testing set)
- **Split**: 70% training, 15% validation, 15% testing

### Model Configuration
- **Classifier**: EfficientNet-B4 with pretrained ImageNet weights
- **Segmentation Encoders**: ResNet34 with ImageNet initialization
- **Normalization**: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Input Channels**: 3 (RGB)
- **Output Classes**: 1 (binary mask for segmentation)

### Training Hyperparameters
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-5
- **Epochs**: 30
- **Optimizer**: Adam (default in trainers)
- **Loss**: Focal/Tversky (for segmentation imbalance)

---

## 📋 Technology Stack
- **Deep Learning**: PyTorch 2.0+ with CUDA support
- **Segmentation**: Segmentation-Models-Pytorch (SMP)
- **Augmentation**: Albumentations (geometric + color transforms)
- **Computer Vision**: OpenCV, Pillow, scikit-image
- **Web Framework**: Streamlit with custom CSS
- **Geo-Mapping**: Folium + Streamlit-Folium
- **Streaming**: Pathway (Linux) + custom folder monitoring (Windows/Mac)
- **Reporting**: FPDF2, Piexif (EXIF extraction)
- **Data Science**: NumPy, Pandas, Scikit-learn

---

## 📈 Performance Summary (v3.9.2)

### Classification
- **Trained on**: AerialWaste binary labels
- **Backbone**: EfficientNet-B4
- **Inference Speed**: 
  - CPU: ~1-3 seconds per image
  - GPU (CUDA): <150ms per image (with TTA)

### Segmentation
- **FPN**: IoU = 0.207, Dice = 0.314
- **UNet++**: IoU = 0.201, Dice = 0.311  
- **DeepLabV3+**: IoU = 0.193, Dice = 0.297

### Ensemble Approach
- Predictions from multiple architectures are combined via voting
- Test-Time Augmentation (5 views) improves stability
- Severity scoring aggregates multiple signals into actionable index

---

## 🔧 Advanced Features

### Real-Time Streaming (Linux Only)
Uses **Pathway Framework** for sub-second latency incremental processing:
```bash
python src/streaming/pathway_pipeline.py
```
Monitors `data/stream_incoming/` for new satellite images and outputs to `outputs/pathway/live_events.csv`

**Note**: Pathway requires Linux environment. Use `folder_stream.py` for Windows/Mac testing.

### Multilingual UI
Supports English, Hindi, and Telugu through `translations.py` localization module.

### Theme-Aware Glassmorphism
Automatic Light/Dark mode reactivity via inline CSS. Respects Streamlit's native theme settings.

### Custom EXIF Extraction
Automatically retrieves GPS coordinates from image metadata for precise geolocation on maps.

---

## 🏛️ Repository Maintenance

### Adding New Dataset Splits
1. Place images in `data/raw/AerialWaste/images/images{N}/`
2. Update metadata JSON files with annotations
3. Modify `config.py` to include new image directory

### Training New Models
1. Edit hyperparameters in `config.py`
2. Run respective trainer: `train_classifier.py` or `train_advanced_seg.py`
3. Trained weights saved to `outputs/checkpoints/best_*.pt`

### Model Evaluation
- Classification metrics: Accuracy, Precision, Recall, F1
- Segmentation metrics: IoU, Dice, Sensitivity, Specificity

---

## 📝 Citation & Attribution
**Satellite-Digital-Twin v3.9.2** — Hack For Green Bharat National Competition Submission

*Building the Digital Twin of a Clean, Waste-Free India through AI-Powered Satellite Intelligence* 🛰️🌿

---

## 📧 Support
For issues, questions, or feature requests, please refer to the FEATURE_GUIDE.md and IMPACT_SUMMARY.md documentation files.
