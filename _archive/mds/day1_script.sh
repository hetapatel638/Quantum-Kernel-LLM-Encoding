#!/bin/bash
# Day 1: Quick test with small dataset

echo "Day 1 Quick Test"
echo "Testing AQED-Hybrid on MNIST (500 samples)"

python main.py \
    --mode single \
    --dataset mnist \
    --template linear \
    --n_pca 10 \
    --n_train 500 \
    --n_test 100 \
    --mock_llm

echo "Test complete! Check results/ folder"