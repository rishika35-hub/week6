# %% [markdown]
# # Colab: PaddleOCR End-to-End Quickstart
# This notebook installs PaddleOCR, downloads COCO-Text (example), converts labels, crops rec training data, and runs a quick smoke train.
# Set runtime to GPU!

# %% [code]
# --- Install dependencies ---
!pip install --upgrade pip
!pip install kaggle
# Install PaddleOCR (v3.2.0 tag) - might take a minute
!pip install git+https://github.com/PaddlePaddle/PaddleOCR.git@v3.2.0
# Install paddlepaddle GPU - choose wheel appropriate for runtime runtime; use CPU if GPU mismatched
!pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/stable.html

# %% [code]
# --- Download COCO-Text (kaggle) ---
# Put your kaggle.json into /root/.kaggle/kaggle.json or connect via Colab UI.
!mkdir -p /root/.kaggle
# If you have kaggle.json uploaded to session storage, copy it manually. Otherwise use UI to upload.
# Example:
# from google.colab import files
# files.upload()
# Then move the kaggle.json to /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json || true
!kaggle datasets download -d c7934597/cocotext-v20 -p /content --unzip

# %% [code]
# --- Place scripts into working dir ---
# Create scripts directory and write conversion scripts (copy contents from repo)
!mkdir -p /content/scripts
# For brevity, assume you've pushed the script files into repo and will git clone. Alternatively, paste scripts manually.

# %% [code]
# Convert COCO-Text to PP-OCR labels (example invocation)
!python /content/scripts/convert_cocotext_to_ppocr.py --coco_json /content/annotations.json \
    --images_dir /content/images --out_dir /content/data/det_labels

# %% [code]
# Crop recognition samples
!python /content/scripts/crop_recognition_data.py --det_label_dir /content/data/det_labels \
    --images_dir /content/images --out_crops_dir /content/data/rec_crops --out_label_file /content/data/rec_labels.txt --max_count 2000

# %% [code]
# Quick train (smoke) - pointing at a lightweight config
# You must have PaddleOCR repo available in /content/PaddleOCR (git clone if not)
!git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git /content/PaddleOCR || true
# Replace a config path below if you store a custom config in configs/
!python /content/PaddleOCR/tools/train.py -c /content/PaddleOCR/configs/det/det_db_mobile_v2.0.yml --use_vdl True --save_dir /content/results

# %% [code]
# Quick inference visualization with PaddleOCR python API
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # download models on first run
img_path = '/content/images/000000000785.jpg'
result = ocr.ocr(img_path, cls=True)
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores)
im_show.save('/content/sample_preds.jpg')
print("Saved /content/sample_preds.jpg")
