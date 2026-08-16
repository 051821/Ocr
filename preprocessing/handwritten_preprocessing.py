"""
preprocessing/handwritten_preprocessing.py

Preprocessing for the handwritten model served off EC2 (a vLLM chat/completions
endpoint expecting a base64 data URL). Its size cap (HANDWRITTEN_MAX_DIMENSION
= 720px) is intentionally much smaller than Paddle's 2000px -- the VLM is far
more sensitive to image token count / context length than a dedicated OCR
detector is, so resizing it as large as the printed pipeline would blow up
latency and cost for no accuracy gain.
"""
import base64
import io
import mimetypes

from PIL import Image

import config


def encode_image_b64(raw_bytes, file_name):
    """Returns (base64_str, mime_type) resized to HANDWRITTEN_MAX_DIMENSION."""
    mime, _ = mimetypes.guess_type(file_name)
    if mime not in ("image/jpeg", "image/png"):
        raise ValueError(f"unsupported image type: {mime}")

    with Image.open(io.BytesIO(raw_bytes)) as im:
        if max(im.size) > config.HANDWRITTEN_MAX_DIMENSION:
            im = im.copy()
            im.thumbnail((config.HANDWRITTEN_MAX_DIMENSION, config.HANDWRITTEN_MAX_DIMENSION), Image.LANCZOS)
        buf = io.BytesIO()
        fmt = "PNG" if mime == "image/png" else "JPEG"
        if fmt == "JPEG":
            im.convert("RGB").save(buf, format=fmt)
        else:
            im.save(buf, format=fmt)
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return image_b64, mime
