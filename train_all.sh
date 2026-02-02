#!/bin/bash

# Train All Models Script
# Usage: ./train_all.sh

echo "================================================================================"
echo "                    MOLECULAR COMMUNICATION - TRAINING SUITE"
echo "================================================================================"
echo ""
echo "This script will train all 4 models sequentially:"
echo "  1. Feature MLP    (35 physics features → position)"
echo "  2. Distance MLP   (20 time features → distance)"
echo "  3. CNN            (100×100 heatmap → position)"
echo "  4. DeepSets       (raw molecule data → position)"
echo ""
echo "Estimated total time: ~40-50 minutes on CPU"
echo "================================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "data/molecular_comm_dataset.mat" ]; then
    echo "❌ Error: Dataset not found!"
    echo "Make sure you're in the project root directory."
    echo "Expected: data/molecular_comm_dataset.mat"
    exit 1
fi

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

echo "📁 Log files will be saved to: $LOG_DIR"
echo ""

# Function to train a model
train_model() {
    local model_name=$1
    local model_path=$2
    local log_file="${LOG_DIR}/${model_name}_${TIMESTAMP}.log"
    
    echo "================================================================================"
    echo "🚀 Training: $model_name"
    echo "================================================================================"
    echo "Started at: $(date)"
    echo "Log file: $log_file"
    echo ""
    
    python -m "$model_path" 2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ $model_name training completed successfully!"
        echo "Finished at: $(date)"
    else
        echo ""
        echo "❌ $model_name training failed!"
        echo "Check log file: $log_file"
        return 1
    fi
    echo ""
}

# Start timer
START_TIME=$(date +%s)

# Train each model
train_model "Feature MLP" "models.feature_mlp.train" || exit 1
train_model "Distance MLP" "models.distance_mlp.train" || exit 1
train_model "CNN" "models.cnn.train" || exit 1
train_model "DeepSets" "models.deepsets.train" || exit 1

# Run comparison
echo "================================================================================"
echo "📊 Running Model Comparison"
echo "================================================================================"
python analysis/compare_models.py 2>&1 | tee "${LOG_DIR}/comparison_${TIMESTAMP}.log"

# End timer
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "================================================================================"
echo "                           🎉 ALL TRAINING COMPLETE!"
echo "================================================================================"
echo ""
echo "⏱️  Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "📊 Results Summary:"
echo "  • Feature MLP:   outputs/feature_mlp/model.pt"
echo "  • Distance MLP:  outputs/distance_mlp/model.pt"
echo "  • CNN:           outputs/cnn/model.pt"
echo "  • DeepSets:      outputs/deepsets/model.pt"
echo "  • Comparison:    outputs/comparison/"
echo ""
echo "📋 Log files saved to: $LOG_DIR"
echo ""
echo "Next steps:"
echo "  1. Review training plots in outputs/*/*.png"
echo "  2. Check comparison results in outputs/comparison/"
echo "  3. Test individual predictions with predict.py scripts"
echo "  4. Prepare for presentation using README.md"
echo ""
echo "================================================================================"
