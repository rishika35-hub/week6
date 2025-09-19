#!/usr/bin/env python3
"""
crop_recognition_data.py
Read detection label files (PaddleOCR format) and crop region images for recognition training.

Generates:
- cropped images in out_crops_dir
- a tab-separated label file out_label_file where each line:
  relative/path/to/crop.jpg<TAB>transcription

Usage:
python crop_recognition_data.py --det_label_dir data/det_labels --images_dir data/raw/cocotext/images \
    --out_crops_dir data/rec_crops --out_label_file data/rec_labels.txt --max_count 50000
"""
import os
import argparse
from PIL import Image
from tqdm import tqdm

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parse_line(line):
    # expected: x1,y1,x2,y2,x3,y3,x4,y4,transcription
    parts = line.strip().split(',')
    if len(parts) < 9:
        return None, None
    coords = list(map(int, parts[:8]))
    text = ','.join(parts[8:]).strip()
    return coords, text

def crop_from_poly(img, poly):
    xs = poly[0::2]; ys = poly[1::2]
    xmin, xmax = max(0,min(xs)), min(img.width, max(xs))
    ymin, ymax = max(0,min(ys)), min(img.height, max(ys))
    if xmax - xmin < 3 or ymax - ymin < 3:
        return None
    return img.crop((xmin, ymin, xmax, ymax))

def main(det_label_dir, images_dir, out_crops_dir, out_label_file, max_count=None):
    ensure_dir(out_crops_dir)
    labels = []
    count = 0
    files = sorted(os.listdir(det_label_dir))
    for fname in tqdm(files, desc='Label files'):
        if not fname.lower().endswith('.txt'):
            continue
        stem = os.path.splitext(fname)[0]
        img_path_jpg = os.path.join(images_dir, stem + '.jpg')
        img_path_png = os.path.join(images_dir, stem + '.png')
        img_path = img_path_jpg if os.path.exists(img_path_jpg) else (img_path_png if os.path.exists(img_path_png) else None)
        if img_path is None:
            continue
        img = Image.open(img_path).convert('RGB')
        with open(os.path.join(det_label_dir, fname), 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                coords, text = parse_line(line)
                if not coords or not text or text == '###':
                    continue
                crop = crop_from_poly(img, coords)
                if crop is None:
                    continue
                out_name = f"{stem}_{i}.jpg"
                out_path = os.path.join(out_crops_dir, out_name)
                crop.save(out_path, quality=95)
                # Store relative path to keep repo portable
                labels.append(f"{os.path.join('data','rec_crops',out_name)}\t{text}")
                count += 1
                if max_count and count >= max_count:
                    break
        if max_count and count >= max_count:
            break
    # write label file
    with open(out_label_file, 'w', encoding='utf-8') as lf:
        lf.write("\n".join(labels))
    print(f"Saved {count} crops to {out_crops_dir} and labels to {out_label_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_label_dir', required=True)
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--out_crops_dir', required=True)
    parser.add_argument('--out_label_file', required=True)
    parser.add_argument('--max_count', type=int, default=None)
    args = parser.parse_args()
    main(args.det_label_dir, args.images_dir, args.out_crops_dir, args.out_label_file, args.max_count)
