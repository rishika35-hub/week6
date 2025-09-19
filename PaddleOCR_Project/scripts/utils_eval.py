#!/usr/bin/env python3
"""
utils_eval.py
Small evaluation utilities: IoU-based detection F-score and normalized edit distance for recognition.
These are minimal helpers — for full evaluation prefer PaddleOCR's `tools/` evaluation scripts.
"""
import numpy as np
from difflib import SequenceMatcher

def iou_poly(boxA, boxB):
    # Simplified polygon IoU using bounding boxes of polygons
    ax = boxA[0::2]; ay = boxA[1::2]
    bx = boxB[0::2]; by = boxB[1::2]
    a_minx, a_maxx = min(ax), max(ax)
    a_miny, a_maxy = min(ay), max(ay)
    b_minx, b_maxx = min(bx), max(bx)
    b_miny, b_maxy = min(by), max(by)
    inter_w = max(0, min(a_maxx, b_maxx) - max(a_minx, b_minx))
    inter_h = max(0, min(a_maxy, b_maxy) - max(a_miny, b_miny))
    inter = inter_w * inter_h
    areaA = (a_maxx - a_minx) * (a_maxy - a_miny)
    areaB = (b_maxx - b_minx) * (b_maxy - b_miny)
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union

def detection_fscore(gt_list, pred_list, iou_thresh=0.5):
    """
    gt_list: list of polygons (list of 8 ints)
    pred_list: list of polygons
    returns precision, recall, f1
    naive greedy matching
    """
    matched = set()
    tp = 0
    for p in pred_list:
        best_iou = 0
        best_j = -1
        for j, g in enumerate(gt_list):
            if j in matched: continue
            i = iou_poly(g, p)
            if i > best_iou:
                best_iou = i; best_j = j
        if best_iou >= iou_thresh:
            tp += 1
            matched.add(best_j)
    fp = len(pred_list) - tp
    fn = len(gt_list) - tp
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1

def normalized_edit_distance(gt, pred):
    sm = SequenceMatcher(None, gt, pred)
    ratio = sm.ratio()
    # normalized edit distance (1 - similarity)
    return 1.0 - ratio

if __name__ == '__main__':
    # quick smoke
    gt = [[0,0,10,0,10,10,0,10]]
    pred = [[0,0,10,0,10,10,0,10]]
    print("F1:", detection_fscore(gt, pred))
    print("ned:", normalized_edit_distance("hello","h3llo"))
