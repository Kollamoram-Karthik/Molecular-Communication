# Molecular Communication Transmitter Localization

This repository implements multiple machine learning approaches for **transmitter localization** in molecular communication systems. Given molecular absorption data at a spherical receiver, the models predict the 2D position $(x_0, y_0)$ of the transmitter.

## 🎯 Problem Statement

A point transmitter emits molecules that diffuse through 3D space. A spherical absorbing receiver at the origin captures molecules that reach its surface. Given:
- **Absorption times**: When each molecule was absorbed
- **Impact angles**: The angle at which each molecule hit the receiver surface

**Goal**: Predict the transmitter's 2D position $(x_0, y_0)$ in micrometers.

---

## 📁 Repository Structure

```
UGP/
├── data/                           # Data generation (MATLAB)
│   ├── generate_data.m             # Main dataset generator
│   ├── molecular_comm_dataset.mat  # Generated dataset (5000 samples)
│   ├── test_heatmap.m              # Quick test script
│   └── visualize_heatmap.m         # Visualization utility
│
├── models/                         # All ML models
│   ├── feature_mlp/                # Model 1: Hand-crafted features → (x₀, y₀)
│   │   ├── model.py                # Architecture + feature extraction
│   │   ├── train.py                # Training script
│   │   └── predict.py              # Inference script
│   │
│   ├── distance_mlp/               # Model 2: Hand-crafted features → distance
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── cnn/                        # Model 3: Heatmap image → (x₀, y₀)
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   └── deepsets/                   # Model 4: Raw data → (x₀, y₀) [NEW]
│       ├── model.py                # DeepSets architecture
│       ├── train.py
│       └── predict.py
│
├── utils/                          # Shared utilities
│   ├── data_loader.py              # Dataset loading (HDF5/MAT files)
│   └── metrics.py                  # Evaluation metrics
│
├── analysis/                       # Analysis scripts
│   └── compare_models.py           # Side-by-side model comparison
│
├── outputs/                        # Trained model checkpoints
│   ├── feature_mlp/
│   ├── distance_mlp/
│   ├── cnn/
│   └── deepsets/
│
├── requirements.txt
└── README.md
```

---

## 🧠 Models Overview

### Model 1: Feature MLP (Hand-crafted Features → Position)

**Input**: 35 physics-informed features extracted from absorption data
**Output**: $(x_0, y_0)$ coordinates

**Features include**:
- Time statistics: mean, median, std, percentiles, skew, kurtosis
- Angle statistics: circular mean, resultant length, sin/cos moments
- Cross features: time-angle correlation
- Physics estimates: calibrated distance and direction

**Architecture**:
```
Input(35) → [Linear→BN→LeakyReLU→Dropout] × 3 → Linear(2)
           with residual connection
```

### Model 2: Distance MLP (Hand-crafted Features → Distance Only)

**Input**: 20 time-based features (no angles)
**Output**: Euclidean distance $\sqrt{x_0^2 + y_0^2}$

Simpler model that only estimates distance, not direction.

### Model 3: CNN (Heatmap → Position)

**Input**: 100×100 2D histogram of (time, angle) pairs
**Output**: $(x_0, y_0)$ coordinates

**Architecture**:
```
Conv2d(1→16) → MaxPool → Conv2d(16→32) → MaxPool → 
Conv2d(32→64) → MaxPool → GAP → FC(64→32→2)
```

### Model 4: DeepSets (Raw Data → Position) ⭐ NEW

**Input**: Raw $(N_0 \times 2)$ matrix of [time, angle] per molecule
**Output**: $(x_0, y_0)$ coordinates

**Key Innovation**: Learns directly from raw data without hand-crafted features!

**Architecture** (DeepSets):
```
Per-molecule:  φ(x) = Linear(2→64) → ReLU → Linear(64→128) → ReLU → Linear(128→128)
Aggregation:   z = mean(φ(x₁), φ(x₂), ..., φ(xₙ))   [permutation invariant]
Set-level:     ρ(z) = Linear(128→64) → ReLU → Linear(64→32) → ReLU → Linear(32→2)
```

**Why DeepSets?**
- Hand-crafted features (mean, std, percentiles) may lose information
- DeepSets learns what aggregations are useful
- Permutation invariant: order of molecules doesn't matter

---

## 🔬 Physics Background

### Diffusion Model

Molecules undergo 3D Brownian motion:
$$X(t + \Delta t) = X(t) + \sigma \cdot \mathcal{N}(0, 1)$$

where $\sigma = \sqrt{2D\Delta t}$ and $D = 100 \, \mu m^2/s$ is the diffusion coefficient.

### Key Relationships

1. **Mean first passage time** scales with distance squared:
   $$\mathbb{E}[T] \propto \frac{d^2}{D}$$

2. **Impact angle distribution** is biased toward transmitter direction:
   $$\hat{\theta} = \text{atan2}\left(\frac{1}{N_0}\sum_i \sin(\theta_i), \frac{1}{N_0}\sum_i \cos(\theta_i)\right)$$

### Fundamental Accuracy Limit

Due to stochastic diffusion, sub-micrometer accuracy is physically limited. Expected error is ~5-10 μm depending on transmitter distance.

---

## 🚀 Quick Start

### 1. Generate Dataset (MATLAB)
```matlab
cd data
run generate_data.m
% Creates molecular_comm_dataset.mat with 5000 samples
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models
```bash
# Train all models
python -m models.feature_mlp.train
python -m models.distance_mlp.train
python -m models.cnn.train
python -m models.deepsets.train
```

### 4. Compare Models
```bash
python analysis/compare_models.py
```

### 5. Make Predictions
```bash
# Feature MLP
python -m models.feature_mlp.predict --num_samples 10

# DeepSets
python -m models.deepsets.predict --evaluate

# Custom input
python -m models.feature_mlp.predict --times "1.5,2.3,3.1" --angles "0.5,-0.3,1.2"
```

---

## 📊 Expected Results

| Model | Mean Error (μm) | Median Error (μm) | Within 5μm | Within 10μm |
|-------|-----------------|-------------------|------------|-------------|
| Feature MLP | ~8-12 | ~7-10 | ~30-40% | ~55-65% |
| CNN | ~10-15 | ~8-12 | ~25-35% | ~50-60% |
| DeepSets | ~8-12 | ~7-10 | ~30-40% | ~55-65% |

*Results depend on dataset size and training hyperparameters.*

---

## 🔮 Future Work Ideas

1. **Attention Mechanisms**: Add self-attention to DeepSets (Set Transformer)
2. **Uncertainty Quantification**: Predict confidence intervals
3. **3D Localization**: Extend to full $(x_0, y_0, z_0)$ prediction
4. **Multi-receiver Setup**: Use multiple receivers for triangulation
5. **Time-series Models**: Treat absorption as a point process (Hawkes process)
6. **Physics-Informed Neural Networks**: Incorporate diffusion PDE as constraint
7. **Larger Datasets**: More samples with varied parameters
8. **Real Experimental Data**: Validate with physical experiments

---

## 📚 References

1. **DeepSets**: Zaheer et al., "Deep Sets", NeurIPS 2017
2. **Molecular Communication**: Farsad et al., "A Comprehensive Survey of Recent Advancements in Molecular Communication", IEEE Communications Surveys & Tutorials, 2016
3. **Diffusion Theory**: Berg, "Random Walks in Biology", Princeton University Press, 1993

---

## 📄 License

MIT License

---

## 👤 Author

Karthik - IIT Delhi
