# Quick Reference Card

## 🚀 Train All 4 Models

```bash
cd /Users/karthik/Developer/UGP

# Model 1: Feature MLP (35 physics features → position)
python -m models.feature_mlp.train

# Model 2: Distance MLP (20 time features → distance)  
python -m models.distance_mlp.train

# Model 3: CNN (100×100 heatmap → position)
python -m models.cnn.train

# Model 4: DeepSets (raw time,angle pairs → position) [NEW]
python -m models.deepsets.train

# Compare all models
python analysis/compare_models.py
```

---

## 📊 Saved Model Locations

| Model | Saved To | Size |
|-------|----------|------|
| Feature MLP | `outputs/feature_mlp/model.pt` | ~100KB |
| Distance MLP | `outputs/distance_mlp/model.pt` | ~50KB |
| CNN | `outputs/cnn/model.pt` | ~150KB |
| DeepSets | `outputs/deepsets/model.pt` | ~120KB |

---

## 🧪 Test/Predict

```bash
# Full evaluation
python -m models.feature_mlp.predict --num_samples 10
python -m models.distance_mlp.predict --evaluate
python -m models.cnn.predict --evaluate
python -m models.deepsets.predict --evaluate
```

---

## 📁 Expected Outputs

```
outputs/
├── feature_mlp/
│   ├── model.pt                 ← Trained weights
│   └── training_results.png     ← Loss curves + predictions
├── distance_mlp/
│   ├── model.pt
│   └── training_results.png
├── cnn/
│   ├── model.pt
│   ├── best_model.pt           ← Best checkpoint
│   └── training_results.png
├── deepsets/
│   ├── model.pt
│   └── training_results.png
└── comparison/
    ├── metrics_table.txt        ← Performance table
    └── comparison_plots.png     ← Side-by-side comparison
```

---

## ⚡ Key Differences

| Model | Input | Features | Parameters |
|-------|-------|----------|-----------|
| **Feature MLP** | Times, Angles | 35 hand-crafted | ~17k |
| **Distance MLP** | Times only | 20 time stats | ~2k |
| **CNN** | 100×100 heatmap | Learned filters | ~34k |
| **DeepSets** | Raw (time, angle) | Learned φ & ρ | ~25k |

---

## 🎯 Why DeepSets is Different

- **No feature engineering** - learns from raw data
- **Permutation invariant** - molecule order doesn't matter  
- **Learns what matters** - discovers optimal aggregations
- **Mathematical guarantee** - universal approximation for sets

---

## ⏱️ Training Time (CPU)

- Feature MLP: ~10-15 min
- Distance MLP: ~5-8 min  
- CNN: ~8-12 min
- DeepSets: ~12-18 min

**Total: ~40-50 minutes for all 4 models**

---

## 🐛 Common Issues

**Import errors?**  
→ Use `-m` flag: `python -m models.feature_mlp.train`

**Dataset not found?**  
→ Check: `ls data/molecular_comm_dataset.mat`

**Model doesn't load?**  
→ Train it first before prediction

---

## 📖 More Information

- **Full instructions:** [TRAINING_INSTRUCTIONS.md](TRAINING_INSTRUCTIONS.md)
- **Project background:** [README.md](README.md)
- **Code details:** Check each model's folder

---

## 🎓 For Your Presentation

Key points to remember:

1. **Physics:** 3D Brownian diffusion, D=100 μm²/s
2. **Problem:** Predict transmitter position from arrival times/angles
3. **Challenge:** Information loss (can't see full trajectory)
4. **4 Approaches:** Engineered features, time-only, heatmaps, learned representations
5. **DeepSets advantage:** No manual feature design, learns what matters
6. **Expected accuracy:** ~5-10 μm error (limited by physics)

**Questions they might ask:**
- Why not perfect accuracy? → Diffusion is stochastic, arrival order doesn't contain full position info
- Why DeepSets? → Molecules are unordered set, permutation invariance is natural
- What's the bottleneck? → Information theoretic limit from random diffusion

Good luck! 🚀
