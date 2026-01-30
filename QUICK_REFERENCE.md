# Quick Reference: Heatmap Generation & Usage

## 🚀 Quick Start

### 1. Test the Implementation (Recommended First)
```matlab
% In MATLAB - Run quick test with 10 samples (~30 seconds)
>> test_heatmap
```

### 2. Generate Full Dataset with Heatmaps
```matlab
% In MATLAB - Generate 5000 samples with heatmaps (~30 minutes)
>> main
```

### 3. Visualize Results
```matlab
% In MATLAB
>> visualize_heatmap
```

```bash
# In Python
python visualize_heatmap_python.py
```

## 📊 What Was Implemented

### Modified Files
- [main.m](main.m) - Added heatmap generation (100×100 resolution)
- [ml_model/data_loader.py](ml_model/data_loader.py) - Updated to load heatmaps

### New Files
- [test_heatmap.m](test_heatmap.m) - Quick test script
- [visualize_heatmap.m](visualize_heatmap.m) - MATLAB visualization
- [visualize_heatmap_python.py](visualize_heatmap_python.py) - Python visualization
- [ml_model/heatmap_dataset.py](ml_model/heatmap_dataset.py) - PyTorch Dataset class template
- [HEATMAP_IMPLEMENTATION.md](HEATMAP_IMPLEMENTATION.md) - Detailed documentation

## 🔧 How It Works

### Heatmap Structure
```
Heatmap: 100 × 100 matrix
- Rows (100): Time bins from 0 to 100 seconds
- Columns (100): Angle bins from -π to π radians
- Values: Integer counts of molecules in each bin
```

### Example
```matlab
% Each sample now has:
sample = struct(
    'x0', 45.2,              % μm
    'y0', 67.8,              % μm
    'N0', 653,               % molecules absorbed
    'heatmap', [100×100]     % Fixed-size image
);
```

## 🐍 Python Usage

### Load Data for Feature-Based Models (Current)
```python
from ml_model.data_loader import load_dataset

# Don't load heatmaps (faster, less memory)
samples = load_dataset('molecular_comm_dataset.mat', load_heatmaps=False)
```

### Load Data for CNN Models (Future)
```python
from ml_model.data_loader import load_dataset

# Load heatmaps for CNN training
samples = load_dataset('molecular_comm_dataset.mat', load_heatmaps=True)

# Access heatmap
heatmap = samples[0]['heatmap']  # Shape: (100, 100)
```

### Use PyTorch DataLoader
```python
from ml_model.heatmap_dataset import create_dataloaders

# Create dataloaders for CNN training
train_loader, val_loader, test_loader = create_dataloaders(
    'molecular_comm_dataset.mat',
    batch_size=32,
    normalize=True,
    log_transform=True
)

# Training loop
for heatmaps, targets in train_loader:
    # heatmaps: (batch, 1, 100, 100)
    # targets: (batch, 2) - (x0, y0)
    predictions = model(heatmaps)
    loss = criterion(predictions, targets)
    ...
```

## 📈 Expected Results

### Performance
- **Test (10 samples):** ~30 seconds
- **Full (5000 samples):** ~25-30 minutes
- **Heatmap overhead:** <0.1% (very fast)

### Memory Usage
- **Per heatmap:** 78 KB
- **5000 heatmaps:** ~390 MB
- **Total dataset:** ~600-700 MB

### Data Quality
- **Sparsity:** ~90-95% zeros (sparse heatmaps)
- **Dynamic range:** 0 to ~50 molecules per bin
- **Non-zero bins:** ~5-10% (depends on N0)

## 🎯 Next Steps for CNN Development

### 1. Verify Dataset
```bash
# Run test first
matlab -batch "test_heatmap"

# Generate full dataset
matlab -batch "main"

# Visualize
python visualize_heatmap_python.py
```

### 2. Design CNN Architecture
Options:
- Simple CNN (3-4 conv layers)
- ResNet-18 (pretrained or from scratch)
- Custom architecture

### 3. Training Pipeline
```python
# Example CNN model
import torch.nn as nn

class HeatmapCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 2)  # Output: (x0, y0)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 4. Compare with Feature-Based Model
- Train both models on same splits
- Compare MAE, accuracy within 5μm
- Visualize prediction errors
- Ensemble if beneficial

## 🔍 Troubleshooting

### Issue: "Undefined function 'generate_heatmap'"
**Solution:** Make sure you're running the updated [main.m](main.m) that includes the function definition at the end.

### Issue: Python can't load heatmaps
**Solution:** 
1. Check dataset was generated with updated [main.m](main.m)
2. Use `load_heatmaps=True` in load_dataset()
3. Verify 'heatmap' field exists in .mat file

### Issue: Memory error
**Solution:** 
1. Reduce batch_size in DataLoader
2. Use fewer num_workers
3. Consider lower resolution (50×50)

### Issue: Sparse heatmaps (too many zeros)
**Expected behavior:** Molecular diffusion is sparse. Solutions:
1. Use log_transform=True (recommended)
2. Normalize appropriately
3. Consider adaptive binning (future work)

## 📁 File Organization

```
UGP/
├── main.m                              # Modified - generates heatmaps
├── test_heatmap.m                      # New - quick test
├── visualize_heatmap.m                 # New - MATLAB viz
├── visualize_heatmap_python.py         # New - Python viz
├── molecular_comm_dataset.mat          # Updated - includes heatmaps
├── HEATMAP_IMPLEMENTATION.md           # New - detailed docs
├── QUICK_REFERENCE.md                  # This file
├── ml_model/
│   ├── data_loader.py                  # Modified - loads heatmaps
│   └── heatmap_dataset.py              # New - PyTorch Dataset
└── outputs/
    ├── heatmap_visualization.png       # Generated
    └── dataloader_test.png             # Generated
```

## 💡 Tips

1. **Start with test_heatmap.m** to verify everything works
2. **Use log_transform=True** for CNN training (handles dynamic range)
3. **Normalize heatmaps** before feeding to CNN
4. **Monitor sparsity** - if >98%, consider rebinning
5. **Compare with features** - heatmaps may not always be better
6. **Visualize predictions** - see where model struggles

## 📚 Documentation

- Full details: [HEATMAP_IMPLEMENTATION.md](HEATMAP_IMPLEMENTATION.md)
- Main dataset script: [main.m](main.m)
- Data loader: [ml_model/data_loader.py](ml_model/data_loader.py)
- PyTorch Dataset: [ml_model/heatmap_dataset.py](ml_model/heatmap_dataset.py)

## ✅ Verification Checklist

Before running full dataset generation:

- [ ] Run `test_heatmap.m` successfully
- [ ] Verify heatmap visualization looks reasonable
- [ ] Check memory usage is acceptable
- [ ] Confirm timing is ~3 sec/sample
- [ ] Review HEATMAP_IMPLEMENTATION.md

After generation:

- [ ] Check file size (~600-700 MB)
- [ ] Run `visualize_heatmap.m`
- [ ] Run `visualize_heatmap_python.py`
- [ ] Test PyTorch DataLoader (if using)
- [ ] Verify heatmap shapes (100×100)

---

**Ready to generate the full dataset?**
```matlab
>> main
```

**Questions?** Check [HEATMAP_IMPLEMENTATION.md](HEATMAP_IMPLEMENTATION.md)
