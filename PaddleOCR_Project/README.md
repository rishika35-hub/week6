# PaddleOCR_End-to-End_Project

Purpose: Complete, copy-paste-ready pipeline to study, convert datasets, train, evaluate, and visualize PP-OCR (PaddleOCR) detection + recognition models. Supports PP-OCRv3 → PP-OCRv5 pipelines.

## Quick start
1. Create structure:
   ```bash
   python scripts/prepare_ppocr_structure.py --root PaddleOCR_Project
   cd PaddleOCR_Project
Place raw dataset under data/raw (e.g., COCO-Text images + annotations.json).

Convert detection labels
python scripts/convert_cocotext_to_ppocr.py --coco_json data/raw/annotations.json --images_dir data/raw/images --out_dir data/det_labels
Crop for recognition:


python scripts/crop_recognition_data.py --det_label_dir data/det_labels --images_dir data/raw/images \
    --out_crops_dir data/rec_crops --out_label_file data/rec_labels.txt --max_count 10000
Use the included notebook (notebooks/Colab_PaddleOCR.py) or run PaddleOCR training:


python PaddleOCR/tools/train.py -c configs/det_db_mobile_v2.0_custom.yml
Structure
(See project root layout; all scripts under scripts/, notebooks under notebooks/, configs under configs/, data under data/.)

Notes & Caveats
The scripts are pragmatic helpers, not full dataset ETL for every edge case. COCO-Text and ICDAR variants contain format quirks; validate randomly sampled outputs.

Use the official PaddleOCR configs for production runs. Custom .yml files here are small examples for quick experimentation.

For full reproducibility on Colab/GCP/AWS, pin the correct paddlepaddle-gpu wheel relative to CUDA and GPU driver.

r
Copy code

---

# `docs/report.tex`
```latex
\documentclass{article}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}
\begin{document}
\title{PaddleOCR End-to-End Study and Training}
\author{Your Name}
\date{\today}
\maketitle

\begin{abstract}
This report documents architecture, datasets, training, and evaluation of PP-OCR pipelines (v3→v5).
\end{abstract}

\section{Architecture}
Describe detector (DB), cls, recognizer (CRNN / PP-OCRv5). Include diagrams in ./docs/diagrams.

\section{Datasets}
Enumerate COCO-Text v2.0, ICDAR 2015, ICDAR 2019 MLT, LSVT, RCTW-17, MTWI.

\section{Training}
Hyperparameters, augmentation, mixed precision usage.

\section{Results}
Precision / Recall / F-measure tables for detection. Accuracy / NED for recognition.

\section{Conclusions}
Lessons learned and next steps.

\end{document}
.gitignore

__pycache__/
*.pyc
data/
results/
*.pth
.vscode/
.ipynb_checkpoints
.env
LICENSE (MIT)
text
Copy code
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
...
[standard MIT text; paste the full MIT license body here]
Small sample data file (example)
Place this example under data/det_labels/000000000785.txt to check the pipeline quickly:

10,20,110,20,110,60,10,60,Hello
120,30,260,30,260,70,120,70,World
And a sample data/rec_labels.txt line:

data/rec_crops/000000000785_0.jpg	Hello
data/rec_crops/000000000785_1.jpg	World