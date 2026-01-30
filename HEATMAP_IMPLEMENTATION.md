# Heatmap Implementation Summary

## Overview
Implemented 2D histogram (heatmap) generation for molecular communication dataset to enable CNN-based position prediction. Each heatmap is a fixed-size image where pixel intensity represents molecule count at specific time-angle coordinates.

## Implementation Details

### 1. MATLAB Changes ([main.m](main.m))

**Added Parameters:**
```matlab
time_bins = 100;     % High resolution time axis
angle_bins = 100;    % High resolution angle axis
time_min = 0;        % Min time: 0 seconds
time_max = T;        % Max time: 100 seconds  
angle_min = -pi;     % Min angle: -π radians
angle_max = pi;      % Max angle: π radians
```

**Heatmap Generation Function:**
- Uses `histcounts2()` to bin absorption events
- Input: variable-length arrays (absorption_times, impact_angles)
- Output: fixed 100×100 matrix
- Handles empty input (samples with N0=0)

**Dataset Structure Updated:**
```matlab
dataset{i} = struct(
    'x0', x0,
    'y0', y0,
    'distance', distance,
    'N0', N0,
    'absorption_times', times,    % Variable length
    'impact_angles', angles,      % Variable length
    'heatmap', heatmap           % Fixed 100×100 matrix ← NEW
);
```

**Saved Variables:**
Added heatmap metadata to .mat file: `time_bins`, `angle_bins`, `time_min`, `time_max`, `angle_min`, `angle_max`

### 2. Python Data Loader ([ml_model/data_loader.py](ml_model/data_loader.py))

**Updated `load_dataset()` function:**
```python
def load_dataset(mat_path, load_heatmaps=False):
    # New parameter: load_heatmaps
    # When True, includes 'heatmap' field in sample dict
```

**Usage:**
```python
# For feature-based models (current)
samples = load_dataset('dataset.mat', load_heatmaps=False)

# For CNN models (future)
samples = load_dataset('dataset.mat', load_heatmaps=True)
```

### 3. Visualization Tools

**MATLAB Visualization ([visualize_heatmap.m](visualize_heatmap.m)):**
- Displays 4 random heatmaps from dataset
- Shows position, N0, and heatmap intensity
- Memory usage statistics

**Python Visualization ([visualize_heatmap_python.py](visualize_heatmap_python.py)):**
- Loads heatmaps via updated data_loader
- Creates publication-quality plots
- Analyzes sparsity and statistics
- Saves to `outputs/heatmap_visualization.png`

**Quick Test ([test_heatmap.m](test_heatmap.m)):**
- Generates 10 samples for quick verification
- Shows raw data vs heatmap comparison
- Reports timing and memory usage
- Validates implementation before full run

## Heatmap Properties

### Dimensions
- **Shape:** 100 × 100 (time bins × angle bins)
- **Time axis:** 0 to 100 seconds (rows)
- **Angle axis:** -π to π radians (columns)
- **Values:** Integer counts (molecules per bin)

### Memory Usage
- **Per heatmap:** ~78 KB (100×100×8 bytes)
- **5000 samples:** ~390 MB total
- **Sparsity:** ~90-95% zeros (estimated)

### Interpretation
- **Pixel [i, j]:** Count of molecules absorbed at time bin i and angle bin j
- **Bright regions:** High molecular density (multiple molecules)
- **Dark regions:** Few/no molecules
- **Patterns:** Encode both temporal and spatial information

## CNN Training Readiness

### Input Format
```python
# Shape: (batch_size, 1, 100, 100)
# Single channel grayscale image
heatmap_tensor = torch.FloatTensor(sample['heatmap']).unsqueeze(0)
```

### Preprocessing Options
1. **Normalization:** Min-max or Z-score normalization
2. **Log transform:** `log(count + 1)` to handle high dynamic range
3. **Standardization:** Per-dataset mean/std normalization

### Advantages Over Variable-Length Input
✓ Fixed input size for CNNs  
✓ Spatial structure preserved  
✓ Time and angle information combined  
✓ Can use pretrained CNN architectures  
✓ Data augmentation via image transforms  
✓ Potential for transfer learning

## Next Steps

### To Generate Full Dataset
```bash
# In MATLAB
>> main  # Generates 5000 samples with heatmaps
```

### To Verify Implementation
```bash
# Quick test (10 samples, ~30 seconds)
>> test_heatmap

# Full visualization after generation
>> visualize_heatmap

# Python visualization
python visualize_heatmap_python.py
```

### For CNN Development
1. Load data with `load_heatmaps=True`
2. Create PyTorch Dataset class
3. Implement CNN architecture (e.g., ResNet, U-Net)
4. Train to predict (x0, y0) from heatmap images
5. Compare performance with feature-based models

## File Structure
```
UGP/
├── main.m                          # Main dataset generation (MODIFIED)
├── test_heatmap.m                  # Quick test script (NEW)
├── visualize_heatmap.m             # MATLAB visualization (NEW)
├── visualize_heatmap_python.py     # Python visualization (NEW)
├── ml_model/
│   └── data_loader.py              # Updated loader (MODIFIED)
└── outputs/
    └── heatmap_visualization.png   # Generated plots
```

## Technical Notes

### Why 100×100 Resolution?
- **Balance:** Detail vs memory/computation
- **CNN-friendly:** Divisible by 2 (pooling layers)
- **Sufficient:** Captures temporal and angular patterns
- **Scalable:** Can adjust if needed (50×50, 128×128)

### Angle Convention
- Range: **-π to π** (consistent with `atan2()`)
- 0 rad = positive x-axis
- π/2 rad = positive y-axis
- -π rad = negative x-axis

### Time Binning
- Range: **0 to 100 seconds** (full simulation window)
- Most absorption: 1-50 seconds
- Early bins: sparse (molecules haven't reached yet)
- Late bins: sparse (most already absorbed)

### Handling Edge Cases
- **No absorption (N0=0):** Returns zero matrix
- **Few molecules:** Sparse heatmap (many zeros)
- **Many molecules:** Dense heatmap (better signal)

## Performance Expectations

### Generation Time
- Test (10 samples): ~30 seconds
- Full (5000 samples): ~25-30 minutes
- Heatmap generation: <0.1% overhead (very fast)

### File Size
- Current dataset: ~200-300 MB
- With heatmaps: ~600-700 MB (additional 390 MB)
- HDF5 compression helps reduce size

## Questions & Future Work

### Potential Improvements
1. **Adaptive binning:** Higher resolution near peak absorption
2. **Multi-channel:** Separate early/late time periods
3. **Preprocessing:** Log-scale during generation
4. **Compression:** Sparse matrix storage

### CNN Architecture Ideas
- **Simple CNN:** 3-4 conv layers → FC → (x0, y0)
- **ResNet:** Pretrained ResNet-18 with modified head
- **U-Net:** Encoder-decoder for feature extraction
- **Ensemble:** Combine CNN + feature-based model

### Evaluation Metrics
- Same as current: MAE, accuracy within 5μm
- Additional: Visualization of prediction errors
- Heatmap reconstruction quality
