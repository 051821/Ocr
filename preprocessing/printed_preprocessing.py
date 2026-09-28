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


def deskew_image(img):
    """Detects document text line tilt angle (-15 to +15 deg) and rotates to make text horizontal."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold & find text contours/edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Dilate to connect text horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect
        if w < h:
            angle = angle - 90 if angle > 0 else angle + 90
        # Normalize to [-45, 45]
        while angle < -45:
            angle += 90
        while angle > 45:
            angle -= 90
        if -15 < angle < 15 and abs(angle) > 0.2:
            angles.append(angle)
            
    if angles:
        median_angle = float(np.median(angles))
        if abs(median_angle) > 0.3:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return img


def enhance_printed_image(img_bgr):
    """
    Preprocesses document image for OCR / Vision models (e.g. Mistral OCR):
    1. Shadow Removal / Illumination Normalization via LAB color space background division.
    2. Adaptive Contrast Enhancement via CLAHE.
    3. Text Sharpening via Unsharp Masking to fix slight blur.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    # Convert BGR to LAB color space to operate on Luminance (L) channel without distorting document colors
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # 1. Shadow Removal / Background Illumination Normalization
    h, w = img_bgr.shape[:2]
    kernel_dim = max(31, min(h, w) // 30 | 1)  # Dynamic odd kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_dim, kernel_dim))
    background = cv2.morphologyEx(l_chan, cv2.MORPH_CLOSE, kernel)

    # Divide L by background to remove shadows & ambient lighting gradients
    l_float = l_chan.astype(np.float32)
    bg_float = np.maximum(background.astype(np.float32), 1.0)
    l_norm = np.clip((l_float / bg_float) * 255.0, 0, 255).astype(np.uint8)

    # 2. CLAHE for adaptive contrast enhancement in shadowed/faint areas
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_norm)

    # Recombine LAB channels and convert back to BGR
    enhanced_lab = cv2.merge((l_enhanced, a_chan, b_chan))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 3. Unsharp Masking to sharpen slightly blurred text strokes
    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), sigmaX=2.0)
    sharpened_bgr = cv2.addWeighted(enhanced_bgr, 1.4, gaussian, -0.4, 0)

    return sharpened_bgr


def preprocess_for_mistral(raw_bytes):
    """
    Full preprocessing pipeline for printed documents before sending to Mistral OCR:
    - Decodes image bytes to BGR numpy array
    - Resizes to max dimension 2000px
    - Deskews tilt angle (-15 to +15 deg)
    - Applies shadow removal, CLAHE contrast enhancement, and unsharp mask sharpening
    - Re-encodes back to JPEG bytes
    """
    img = bytes_to_bgr_array(raw_bytes)
    h, w = img.shape[:2]
    max_dim = getattr(config, "PADDLE_MAX_DIMENSION", 2000)
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    img = deskew_image(img)
    img = enhance_printed_image(img)

    # Re-encode to JPEG bytes
    success, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return raw_bytes
    return buffer.tobytes()


def preprocess_for_paddle(raw_bytes):
    img = bytes_to_bgr_array(raw_bytes)
    h, w = img.shape[:2]
    if max(h, w) > config.PADDLE_MAX_DIMENSION:
        scale = config.PADDLE_MAX_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    img = deskew_image(img)

    # Add a white border so characters at the document edge (common in photos taken at an angle)
    # are not clipped by PaddleOCR's internal detection margin.
    pad = config.PADDLE_EDGE_PADDING
    if pad > 0:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img

