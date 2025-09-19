#!/usr/bin/env python3
"""
convert_cocotext_to_ppocr.py
Convert COCO-Text-style JSON annotations to PaddleOCR detection label files.
Each output file: <image_stem>.txt containing lines:
x1,y1,x2,y2,x3,y3,x4,y4,transcription

Usage:
python convert_cocotext_to_ppocr.py --coco_json data/raw/cocotext/annotations.json \
    --images_dir data/raw/cocotext/images --out_dir data/det_labels
"""
import os
import json
import argparse
from tqdm import tqdm

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def xywh_to_polygon(x,y,w,h):
    return [x,y, x+w,y, x+w,y+h, x,y+h]

def main(coco_json, images_dir, out_dir, min_area=4):
    ensure_dir(out_dir)
    with open(coco_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    imgs = {img['id']: img for img in coco.get('images', [])}
    annotations = {}
    for a in coco.get('annotations', []):
        annotations.setdefault(a['image_id'], []).append(a)

    for imgid, img in tqdm(imgs.items(), desc='Images'):
        fname = img.get('file_name')
        if not fname:
            continue
        stem = os.path.splitext(fname)[0]
        anns = annotations.get(imgid, [])
        lines = []
        for a in anns:
            # prefer utf8_string or text fields
            text = a.get('utf8_string') or a.get('text') or a.get('caption') or ''
            if not text or text.strip()=='':
                continue
            # segmentation or bbox
            poly = None
            if 'segmentation' in a and a['segmentation']:
                seg = a['segmentation']
                # segmentation may be list of lists; pick first
                if isinstance(seg, list):
                    if len(seg)==0:
                        continue
                    if isinstance(seg[0], list):
                        seg = seg[0]
                    poly = [int(round(float(x))) for x in seg[:8]]  # first 4 points
            elif 'polygon' in a and a['polygon']:
                p = a['polygon']
                if isinstance(p, list):
                    poly = [int(round(float(x))) for x in p[:8]]
            elif 'bbox' in a and a['bbox']:
                x,y,w,h = a['bbox']
                poly = xywh_to_polygon(x,y,w,h)
                poly = [int(round(float(x))) for x in poly]
            if not poly or len(poly) < 8:
                continue
            # simple area check
            xs = poly[0::2]; ys = poly[1::2]
            if (max(xs)-min(xs))*(max(ys)-min(ys)) < min_area:
                continue
            poly_str = ','.join(str(int(p)) for p in poly[:8])
            safe_text = text.replace('\n', ' ').replace('\r', ' ').strip()
            # ensure no commas are accidentally introduced inside transcription field (PaddleOCR expects last field to be transcription)
            lines.append(f"{poly_str},{safe_text}")
        if lines:
            out_path = os.path.join(out_dir, f"{stem}.txt")
            with open(out_path, 'w', encoding='utf-8') as outf:
                outf.write("\n".join(lines))
    print("Conversion finished. Labels saved to:", out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_json', required=True)
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    main(args.coco_json, args.images_dir, args.out_dir)
