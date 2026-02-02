# Training and Testing Instructions

## Overview
You have **4 models** ready to train:
1. **Feature MLP** - Uses 35 hand-crafted physics features
2. **Distance MLP** - Uses 20 time-based features to predict distance
3. **CNN** - Uses 2D heatmap images (time×angle)
4. **DeepSets** - Uses raw molecular data (time, angle) pairs [NEW]

---

## Prerequisites

Make sure you're in the project root directory and have the required dependencies:

```bash
cd /Users/karthik/Developer/UGP
pip install -r requirements.txt
```

---

## Training Instructions

### 1. Feature MLP Model

**Train:**
```bash
python -m models.feature_mlp.train
```

**What it does:**
- Extracts 35 physics features from each sample
- Trains a residual MLP network (128→128→64→2)
- Uses data augmentation (4 augmentations per sample)
- Trains for up to 500 epochs with early stopping (patience=80)
- **Saves model to:** `outputs/feature_mlp/model.pt`
- **Saves plots to:** `outputs/feature_mlp/training_results.png`

**Expected output:**
- Training progress bar with loss values
- Test set metrics (MAE, median error, % within 5μm and 10μm)
- Training curves and prediction plots

---

### 2. Distance MLP Model

**Train:**
```bash
python -m models.distance_mlp.train
```

**What it does:**
- Extracts 20 time-based features
- Trains simple MLP (20→64→32→1)
- Predicts distance only, not position
- Trains for up to 300 epochs (patience=50)
- **Saves model to:** `outputs/distance_mlp/model.pt`
- **Saves plots to:** `outputs/distance_mlp/training_results.png`

**Expected output:**
- Test set metrics for distance prediction
- Comparison with physics-based baseline
- Improvement percentage over baseline

---

### 3. CNN Model

**Train:**
```bash
python -m models.cnn.train
```

**What it does:**
- Uses 100×100 heatmap images (time×angle bins)
- Trains 3-layer convolutional network (~34k parameters)
- Applies log-transform and normalization to heatmaps
- Trains for up to 100 epochs (patience=20)
- **Saves model to:** `outputs/cnn/model.pt`
- **Also saves:** `outputs/cnn/best_model.pt` (best validation checkpoint)
- **Saves plots to:** `outputs/cnn/training_results.png`

**Expected output:**
- Model parameter count
- Test set metrics with error distribution histogram
- 3-panel plot: training curves, predictions, error histogram

---

### 4. DeepSets Model (NEW)

**Train:**
```bash
python -m models.deepsets.train
```

**What it does:**
- Uses raw (time, angle) pairs directly (no feature engineering!)
- φ network encodes each molecule independently
- Masked mean aggregation (permutation invariant)
- ρ network predicts position from aggregated representation
- Trains for up to 200 epochs (patience=50)
- **Saves model to:** `outputs/deepsets/model.pt`
- **Saves plots to:** `outputs/deepsets/training_results.png`

**Expected output:**
- Model parameter count
- Test set metrics
- Training curves, predictions, and error distribution

---

## Testing/Prediction Instructions

After training, you can test each model:

### Feature MLP Prediction

**Evaluate on full test set:**
```bash
python -m models.feature_mlp.predict --num_samples 10
```

**Custom times and angles:**
```bash
python -m models.feature_mlp.predict --times "1.5,2.3,3.1" --angles "0.5,-0.3,1.2"
```

**Loads model from:** `outputs/feature_mlp/model.pt`

---

### Distance MLP Prediction

**Evaluate on full test set:**
```bash
python -m models.distance_mlp.predict --evaluate
```

**Random samples:**
```bash
python -m models.distance_mlp.predict --num_samples 10
```

**Custom times:**
```bash
python -m models.distance_mlp.predict --times 1.5 2.3 3.1 4.0
```

**Loads model from:** `outputs/distance_mlp/model.pt`

---

### CNN Prediction

**Evaluate on full test set:**
```bash
python -m models.cnn.predict --evaluate
```

**Random samples:**
```bash
python -m models.cnn.predict --num_samples 10
```

**Loads model from:** `outputs/cnn/model.pt`

---

### DeepSets Prediction

**Evaluate on full test set:**
```bash
python -m models.deepsets.predict --evaluate
```

**Random samples:**
```bash
python -m models.deepsets.predict --num_samples 10
```

**Loads model from:** `outputs/deepsets/model.pt`

---

## Model Comparison

After training all models, compare them side-by-side:

```bash
python analysis/compare_models.py
```

**What it does:**
- Loads all 4 trained models
- Evaluates on the same test set
- Creates comparison table
- Generates comparison plots
- **Saves to:** `outputs/comparison/`

**Output:**
- Metrics table (MAE, median error, accuracy within thresholds)
- Error distribution plots for all models
- Visual comparison

---

## Directory Structure After Training

```
outputs/
├── feature_mlp/
│   ├── model.pt                    ← Trained model
│   └── training_results.png        ← Training plots
├── distance_mlp/
│   ├── model.pt                    ← Trained model
│   └── training_results.png        ← Training plots
├── cnn/
│   ├── model.pt                    ← Trained model (final)
│   ├── best_model.pt               ← Best validation checkpoint
│   └── training_results.png        ← Training plots
├── deepsets/
│   ├── model.pt                    ← Trained model
│   └── training_results.png        ← Training plots
└── comparison/
    ├── metrics_table.txt           ← Performance comparison
    └── comparison_plots.png        ← Visual comparison
```

---

## Training Tips

### GPU Acceleration
All models automatically use GPU if available:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Training Time Estimates (CPU)
- **Feature MLP:** ~10-15 minutes
- **Distance MLP:** ~5-8 minutes
- **CNN:** ~8-12 minutes
- **DeepSets:** ~12-18 minutes

### Memory Requirements
- All models should fit in ~2GB RAM
- Dataset: ~350MB in memory

### If Training Stops Early
This is **expected behavior** due to early stopping:
- Feature MLP: stops if validation loss doesn't improve for 80 epochs
- Distance MLP: stops after 50 epochs without improvement
- CNN: stops after 20 epochs without improvement
- DeepSets: stops after 50 epochs without improvement

---

## Troubleshooting

### Issue: "No module named 'utils'"
**Solution:** Make sure you're running from the project root with `-m` flag:
```bash
cd /Users/karthik/Developer/UGP
python -m models.feature_mlp.train  # ✓ Correct
# NOT: python models/feature_mlp/train.py  # ✗ Wrong
```

### Issue: "FileNotFoundError: molecular_comm_dataset.mat"
**Solution:** Ensure dataset is in the correct location:
```bash
ls data/molecular_comm_dataset.mat  # Should exist
```

### Issue: Model not loading during prediction
**Solution:** Train the model first before running prediction:
```bash
# Train first
python -m models.feature_mlp.train
# Then predict
python -m models.feature_mlp.predict
```

### Issue: CUDA out of memory
**Solution:** Reduce batch size in the training script:
```python
# Edit train.py:
batch_size=32  # Instead of 64
```

---

## Quick Start (Train All Models)

Run all 4 models in sequence:

```bash
# Make sure you're in project root
cd /Users/karthik/Developer/UGP

# Train all models
echo "Training Feature MLP..."
python -m models.feature_mlp.train

echo "Training Distance MLP..."
python -m models.distance_mlp.train

echo "Training CNN..."
python -m models.cnn.train

echo "Training DeepSets..."
python -m models.deepsets.train

# Compare results
echo "Comparing all models..."
python analysis/compare_models.py

echo "Done! Check outputs/ directory for results."
```

---

## What Gets Saved

### Model Files (.pt)
Each model saves:
- Network weights (`state_dict`)
- Normalization parameters (means, stds)
- Target ranges for denormalization
- Physics calibration parameters (where applicable)

### Plots (.png)
Each training script generates:
- Training/validation loss curves
- Prediction vs actual scatter plots
- Error distribution histograms (CNN and DeepSets)

### Metrics
- Test set performance metrics printed to console
- Can redirect to file: `python -m models.feature_mlp.train > train_log.txt`

---

## Next Steps

1. **Train all 4 models** (run the commands above)
2. **Compare results** using `analysis/compare_models.py`
3. **Analyze errors** - which model works best?
4. **Prepare for presentation:**
   - Review [README.md](README.md) for background
   - Understand the physics and mathematics
   - Check "Questions Your Professor Might Ask" section

Good luck with your presentation tomorrow! 🎓
