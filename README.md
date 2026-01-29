# Molecular Communication Transmitter Localization

ML-based localization of a point transmitter using molecular absorption data.

## Setup

```bash
pip install numpy scipy torch matplotlib tqdm h5py
```

## Usage

### 1. Generate Dataset (MATLAB)

Run `main.m` in MATLAB to generate simulation data:
- Default: 5000 samples, 2000 molecules each
- Output: `molecular_comm_dataset.mat`

**Note**: With 5000 samples × 2000 molecules, this takes ~30 minutes.

### 2. Train Model

```bash
python train.py
```

Outputs saved to `outputs/`:
- `physics_model.pt` - trained model
- `training_history.png` - loss curves
- `predictions.png` - test set predictions
- `error_distribution.png` - error analysis

### 3. Evaluate Model

```bash
python evaluate.py
```

Shows test set metrics and sample predictions.

## Model Architecture

**Physics-Informed Neural Network**:
1. **Feature Extraction**: 35 physics-based features from absorption times and impact angles
2. **Neural Network**: 3-layer MLP with residual connections
3. **Output**: Predicted (x0, y0) coordinates

## Expected Performance

| Dataset Size | Molecules | Mean Error | Within 5μm |
|-------------|-----------|------------|------------|
| 1000 | 500 | ~9 μm | ~30% |
| 5000 | 2000 | ~3-5 μm | ~60% |

**Physical limit**: Diffusion is inherently stochastic. Sub-micrometer accuracy requires more data or reduced parameter space.

## Files

```
├── main.m                  # MATLAB simulation (data generation)
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── ml_model/
│   ├── data_loader.py      # Load .mat files
│   ├── physics_model.py    # Model architecture
│   └── trainer.py          # Training utilities
└── outputs/                # Saved models and plots
```
