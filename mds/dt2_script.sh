#!/bin/bash
# Day 3: Full evaluation (run on Colab)

echo "Full Evaluation"
echo "This will take several hours..."

python main.py \
    --mode multi \
    --n_pca 80 \
    --n_train 10000 \
    --n_test 10000

echo "Generating visualizations..."
python main.py --mode visualize

echo "Evaluation complete!"