# %% [markdown]
# # Kaggle Notebook: PaddleOCR detection+recognition quick experiment
# Use Kaggle dataset APIs instead of Kaggle CLI; kernels have dataset mounting options.

# %% [code]
# Install PaddleOCR
!pip install git+https://github.com/PaddlePaddle/PaddleOCR.git@v3.2.0
!pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/stable.html

# %% [code]
# If dataset is attached to the kernel, it will be under /kaggle/input/<dataset-name>
# Example file paths:
IMG_DIR = "/kaggle/input/cocotext/images"
ANNOT = "/kaggle/input/cocotext/annotations.json"

# Copy scripts into /kaggle/working and run conversion
!mkdir -p /kaggle/working/scripts
# assume scripts are present; otherwise paste them into file
!python /kaggle/working/scripts/convert_cocotext_to_ppocr.py --coco_json {ANNOT} --images_dir {IMG_DIR} --out_dir /kaggle/working/data/det_labels

# Crop recognition data, etc. then run a short training using a small subset/config.
