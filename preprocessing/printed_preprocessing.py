"""
preprocessing/printed_preprocessing.py

Preprocessing for PaddleOCR ("printed" stage). Deliberately kept separate
from clip_preprocessing.py and handwritten_preprocessing.py: Paddle wants a
BGR numpy array capped at PADDLE_MAX_DIMENSION (2000px -- keep detail for
dense printed tables/lab reports), not the 720px cap the handwritten model
uses and not CLIP's tiling.
"""
import cv2
import numpy as np

import config


def bytes_to_bgr_array(raw_bytes):
    file_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("empty/undecodable image (download or cv2 decode failed)")
    return img


def preprocess_for_paddle(raw_bytes):
    img = bytes_to_bgr_array(raw_bytes)
    h, w = img.shape[:2]
    if max(h, w) > config.PADDLE_MAX_DIMENSION:
        scale = config.PADDLE_MAX_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img
