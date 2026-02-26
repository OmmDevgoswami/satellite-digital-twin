# Satellite-Digital-Twin: Complete User Guide (v3.9.2)

Welcome to **Satellite-Digital-Twin v3.9.2**, an advanced **AI Digital Twin** engineered for the **Hack For Green Bharat** national competition. This guide explains how to use our state-of-the-art infrastructure monitoring features for detecting and analyzing illegal waste dumps from satellite imagery.

---

## 📖 Table of Contents
1. [Application Overview](#application-overview)
2. [Core Features](#core-features)
3. [Using the Web Interface](#using-the-web-interface)
4. [Understanding Results](#understanding-results)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tips](#performance-tips)

---

## Application Overview

**Satellite-Digital-Twin** provides a unified Streamlit web interface for:
- Upload and analyze satellite images for illegal dump detection
- Visualize detection results with spatial segmentation masks
- Estimate environmental impact (waste tonnage, CO₂, cleanup costs)
- View geolocation data on interactive maps
- Generate professional PDF reports
- Monitor real-time satellite telemetry streams (Linux environments)
- Access multi-language interface (English, Hindi, Telugu)

### System Requirements
- **OS**: Windows, macOS, Linux
- **Python**: 3.10+
- **Disk**: Minimum 2GB (models + dependencies)
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **GPU**: NVIDIA CUDA 11.8+ (optional, improves speed 10-20x)

---

## Core Features

### 1. Single Image Classification & Segmentation

#### Input
- Upload a satellite image (JPG, PNG, WEBP)
- Image is automatically resized to 256×256 pixels
- Supported from local files or camera capture

#### Processing
- **Classification**: Binary detector (Dump vs. No-Dump) using EfficientNet-B4
- **Segmentation**: Pixel-level boundary detection using ensemble (FPN, UNet++, DeepLabV3+)
- **Confidence**: Probability score [0, 1] for dump detection
- **Coverage**: Percentage of image area predicted as dump

#### Output
- Confidence score with visual indicator
- Segmentation mask overlay on original image
- Severity index (0-100) with color-coded risk level
- Environmental impact metrics:
  - Estimated waste tonnage
  - CO₂ equivalent emissions
  - Cleanup cost in INR

#### Sidebar Options
- **Toggle TTA (Test-Time Augmentation)**: Enable for 5-view ensemble voting (slower but more robust)
- **Confidence Threshold**: Adjust classification threshold (default: 0.5)
- **Language Selection**: Switch between EN, HI, TE
- **Theme Toggle**: Light/Dark mode (theme-aware CSS)

---

### 2. Spatial Mapping & Geolocation

#### Features
- **Interactive Folium Maps**: Display detected dump sites on high-resolution satellite tiles
- **EXIF Auto-Detection**: Extract GPS coordinates from uploaded image metadata
- **Marker Clustering**: Group nearby detections for regional analysis
- **Heatmap Overlay**: Visualize dump density across monitored areas
- **National Registry**: Teleport to pre-loaded dump locations (Delhi, Mumbai, Bengaluru, etc.)

#### How to Use
1. Upload image with GPS metadata (EXIF data embedded)
2. Click "View on Map" to see location on national satellite tiles
3. Interact with map: zoom, pan, toggle layers
4. Use search to navigate to specific municipalities

---

### 3. Severity Scoring System

#### Scoring Formula
```
Score = 0.4 × P + 0.4 × (C/100) + 0.2 × min(A/10, 1)

Where:
  P = Classifier probability [0, 1]
  C = Coverage percentage [0, 100]
  A = Estimated area in hectares
```

#### Risk Levels
| Level | Score Range | Emoji | Color | Action |
|-------|------------|-------|-------|--------|
| CRITICAL | 75-100 | 🔴 | Red (#c0392b) | Immediate intervention |
| HIGH | 50-75 | 🟠 | Orange (#e67e22) | Priority remediation |
| MEDIUM | 25-50 | 🟡 | Yellow (#f1c40f) | Scheduled cleanup |
| LOW | 0-25 | 🟢 | Green (#27ae60) | Monitoring |

#### Interpretation
- **Higher score** = More severe environmental threat
- **Combines** confidence, spatial extent, and area for holistic risk assessment
- **Sorted results** display highest-risk sites first

---

### 4. Environmental Impact Engine

#### Estimated Metrics
Based on detected dump area and spatial coverage:

**Waste Tonnage**
- Formula: `Effective Area (ha) × 150 Tonnes/ha`
- Assumption: 150 T/Ha average density for mixed waste dumps
- Used to estimate intervention resource requirements

**CO₂ Equivalent Emissions**
- Formula: `Waste Tonnage × 1.2 kg CO₂e/Tonne ÷ 1000`
- Accounts for methane + CO₂ emissions from open dumps
- Aligns with India's climate mitigation targets

**Cleanup Cost Estimate**
- Formula: `Waste Tonnage × ₹2,200/Tonne`
- Uses municipal standard benchmark for India
- Supports budget prioritization for local authorities

#### PDF Report Generation
1. Configure report title and metadata
2. Select dump sites to include
3. Click "Generate Report" to create single-page GIS-encoded PDF
4. Download for municipal submission and archival

---

### 5. Pathway Real-Time Streaming (Linux Only)

#### Prerequisites
- Linux environment with official Pathway package installed
- `data/stream_incoming/` folder prepared

#### How It Works
1. Monitor `data/stream_incoming/` for new satellite images
2. Incrementally process each image with classifier + segmentation
3. Update regional health index in real-time
4. Output detections to `outputs/pathway/live_events.csv`
5. Maintain sub-second latency for continuous monitoring

#### Usage
```bash
python src/streaming/pathway_pipeline.py
```

Monitor output CSV:
```bash
tail -f outputs/pathway/live_events.csv
```

#### Regional Dashboard (Simulated)
- **National Health Index**: Aggregated severity across all zones
- **Zone-wise Risk**: North/South/East/West comparative analysis
- **Alert Thresholds**: Auto-dispatch notifications when zone risk exceeds limit

**Note**: For Windows/macOS testing, use `folder_stream.py` instead.

---

## Using the Web Interface

### Quick Start (60 seconds)

1. **Launch Application**
   ```bash
   streamlit run app/app.py
   ```

2. **Upload Image**
   - Click "Upload Satellite Image" in sidebar
   - Select JPG/PNG/WEBP from filesystem or camera

3. **View Results**
   - Confidence score displayed in metric card
   - Segmentation mask overlaid on image
   - Severity badge (🟢🟡🟠🔴) shows risk level

4. **Understand Impact**
   - Environmental metrics (tonnage, CO₂, cost) shown in expandable section
   - Original + masked images side-by-side comparison

5. **Map Location** (if EXIF available)
   - Click "View on Map" to see geolocation
   - Interact with Folium map

6. **Generate Report** (optional)
   - Click "Generate PDF Report"
   - Download for archival/submission

---

## Understanding Results

### Classification Output
```
Dump Probability: 0.87
Interpretation: 87% confidence this is a dump site
```

### Segmentation Output
```
Coverage: 42%
Interpretation: 42% of image area identified as waste material
```

### Severity Score
```
Score: 68/100 (HIGH 🟠)
Components:
  - Confidence: 0.87 (40%) → 0.348
  - Coverage: 42% (40%) → 0.168
  - Area: 2.5 ha (20%) → 0.05
  - Total: 0.566 → 68/100
```

### Environmental Impact
```
Estimated Area: 1.05 ha
Waste Tonnage: 157.5 T
CO₂ Equivalent: 0.19 MT
Cleanup Cost: ₹346,500
```

---

## Advanced Features

### Test-Time Augmentation (TTA)
Enables robustness to image orientation/tilt:
- **Disabled** (default): Fast inference, 1 prediction
- **Enabled**: 5 geometric views voted + averaged
- **Benefit**: More stable on rotated/skewed satellite tiles
- **Trade-off**: ~5x slower inference

### Ensemble Voting
Multiple segmentation architectures:
- **FPN**: Excels at multi-scale dumps (IoU: 0.207)
- **UNet++**: Sharp boundaries on small patches (IoU: 0.201)
- **DeepLabV3+**: Global context awareness (IoU: 0.193)

Predictions combined via soft voting (average probability maps).

### Batch Processing
For analyzing multiple images:
1. Organize satellite images in folder
2. Use `train_segmentation.py` or custom inference loop
3. Export results to CSV for bulk analysis

### Custom Threshold Calibration
Adjust confidence threshold:
- **Low (0.3)**: Maximize sensitivity, catch more potential sites
- **Medium (0.5)**: Balanced approach (default)
- **High (0.8)**: Maximize precision for confirmed dumps

---

## Troubleshooting

### Issue: "No dump detected" for obvious location
**Solution**:
- Check image is satellite view (not aerial photo)
- Ensure image contains waste material (not vegetation)
- Try enabling TTA for more robust prediction
- Lower confidence threshold temporarily

### Issue: Slow inference (~3s+ on CPU)
**Solution**:
- Disable TTA (default is faster)
- Use GPU if available (10-30x speedup)
- Reduce batch size if processing multiple images

### Issue: EXIF coordinates not extracted
**Solution**:
- Image missing GPS metadata
- Use "Manual Coordinates" input in sidebar
- Re-upload image with EXIF data intact

### Issue: "Pathway not available"
**Solution**:
- Pathway requires Linux environment
- On Windows/macOS, use `folder_stream.py`
- Or submit code in Linux Docker container

### Issue: Out of memory (GPU/CPU)
**Solution**:
- Use GPU with CUDA support
- Enable CPU optimizations in config
- Process images one at a time (avoid batch)

---

## Performance Tips

### Optimize for Speed
1. **GPU Acceleration** (if available)
   - Install CUDA 11.8+ drivers
   - PyTorch will auto-detect CUDA
   - Expect 10-30x speedup vs CPU

2. **Disable Unnecessary Features**
   - Turn off TTA for real-time use
   - Skip PDF generation for batch
   - Use lightweight browser

3. **Batch Processing**
   - Process 10+ images together
   - Amortize model loading time
   - Cache transforms

### Optimize for Accuracy
1. **Enable TTA**
   - Multi-angle voting improves robustness
   - Worth the latency for critical sites

2. **Upload High-Resolution Images**
   - Larger images capture fine details
   - Resized to 256×256, but preprocessing helps

3. **Verify Metadata**
   - Ensure EXIF coordinates are accurate
   - Cross-reference with visual map inspection

### Optimize for Cost (Budget-Aware Deployment)
1. **Run Locally First**
   - No cloud computing costs
   - Useful for prototyping

2. **Batch Cloud Jobs**
   - Submit regions overnight
   - Process thousands simultaneously

3. **Use Lightweight Models**
   - FPN best balance of accuracy/speed
   - Skip DeepLabV3+ for real-time systems

---

## 🌍 Supported Languages

| Language | Code | Status |
|----------|------|--------|
| English | EN | ✅ Full Support |
| Hindi | HI | ✅ Full Support |
| Telugu | TE | ✅ Full Support |

Switch languages in sidebar settings.

---

## 📊 Dataset: AerialWaste

The model is trained on **AerialWaste** dataset:
- **Source**: Hack For Green Bharat official dataset
- **Content**: Satellite images of dump sites and non-dump areas
- **Labels**: Binary classification + polygon segmentation (test set)
- **Regions**: Across India (multiple states/territories)

---

## 💡 Best Practices

1. **Always verify** AI predictions with domain experts
2. **Use severity scores** to prioritize intervention (highest first)
3. **Track historical detections** in CSV exports for trend analysis
4. **Calibrate thresholds** per-region based on local dump characteristics
5. **Generate PDF reports** for official municipal records
6. **Monitor streaming pipeline** latency for real-time deployments

---

## 📧 Support & Feedback

For technical issues or feature requests:
- Check README.md for setup details
- Refer IMPACT_SUMMARY.md for architectural overview
- Review source code documentation in comments
- Test on provided sample images first

---

**Satellite-Digital-Twin v3.9.2 — Empowering Sustainable India with Satellite AI Intelligence** 🛰️🌿
