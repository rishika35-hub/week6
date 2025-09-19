#!/usr/bin/env python3
"""
prepare_ppocr_structure.py
Create the canonical folder structure for the project.

Usage:
python prepare_ppocr_structure.py --root PaddleOCR_Project
"""
import os
import argparse

STRUCT = [
    'scripts',
    'notebooks',
    'configs',
    'data/raw',
    'data/det_labels',
    'data/rec_crops',
    'data/results',
    'results/weights',
    'results/logs',
    'results/predictions',
    'docs/diagrams',
    'docs/latex'
]

def ensure_dirs(root):
    for p in STRUCT:
        full = os.path.join(root, p)
        os.makedirs(full, exist_ok=True)
    # touch readme
    open(os.path.join(root, 'README.md'), 'a').close()
    print(f"Created project skeleton under {root}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='PaddleOCR_Project')
    args = parser.parse_args()
    ensure_dirs(args.root)
