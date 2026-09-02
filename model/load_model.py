"""
model/load_model.py
"""
import gc
import time
import boto3
import torch

import config


# ---------------------------------------------------------------------------
# STAGE 1 -- CLIP
# ---------------------------------------------------------------------------
def load_clip_classifier(device=config.CLIP_DEVICE):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
    model = model.to(device).eval()

    labels = list(config.CLIP_PROMPTS.keys())
    text_tokens = tokenizer([config.CLIP_PROMPTS[l] for l in labels]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return {
        "model": model,
        "preprocess": preprocess,
        "text_features": text_features,
        "labels": labels,
        "device": device,
    }


def unload_clip_classifier(clip_bundle):
    if clip_bundle and clip_bundle.get("model") is not None:
        del clip_bundle["model"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# STAGE 2 -- PaddleOCR (two engines: mobile "en" rec + heavier "server" rec,
# ensembled downstream in model/paddel.py)
# ---------------------------------------------------------------------------
def load_paddle_engines():
    from paddleocr import PaddleOCR

    ocr_en = PaddleOCR(
        text_detection_model_name=config.PADDLE_DET_MODEL,
        text_recognition_model_name=config.PADDLE_REC_MODEL_EN,
        text_det_limit_side_len=config.PADDLE_DET_LIMIT_SIDE_LEN,
        text_det_limit_type="max",
        text_det_thresh=config.PADDLE_DET_THRESH,
        text_det_box_thresh=config.PADDLE_DET_BOX_THRESH,
        text_det_unclip_ratio=config.PADDLE_DET_UNCLIP_RATIO,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        device=config.PADDLE_DEVICE,
    )
    ocr_server = PaddleOCR(
        text_detection_model_name=config.PADDLE_DET_MODEL,
        text_recognition_model_name=config.PADDLE_REC_MODEL_SERVER,
        text_det_limit_side_len=config.PADDLE_DET_LIMIT_SIDE_LEN,
        text_det_limit_type="max",
        text_det_thresh=config.PADDLE_DET_THRESH,
        text_det_box_thresh=config.PADDLE_DET_BOX_THRESH,
        text_det_unclip_ratio=config.PADDLE_DET_UNCLIP_RATIO,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        device=config.PADDLE_DEVICE,
    )
    return ocr_en, ocr_server


def unload_paddle_engines():
    import paddle
    gc.collect()
    if config.PADDLE_DEVICE.startswith("gpu"):
        paddle.device.cuda.empty_cache()


# ---------------------------------------------------------------------------
# STAGE 3 -- EC2 lifecycle for the handwritten model (future_work: automatic
# start/stop around the OCR call). No-op unless AUTO_MANAGE_EC2=true --
# otherwise main.py assumes you already have your SSH tunnel open, matching
# the manual workflow documented at the top of the original handwritten.py.
# ---------------------------------------------------------------------------
def _ec2_client():
    return boto3.client("ec2", region_name=config.EC2_REGION)


def start_ec2_instance():
    if not config.AUTO_MANAGE_EC2:
        print("[ec2] AUTO_MANAGE_EC2=false -- assuming endpoint is already reachable (manual tunnel).")
        return
    if not config.EC2_INSTANCE_ID:
        raise RuntimeError("AUTO_MANAGE_EC2=true but EC2_INSTANCE_ID is not set in .env")

    client = _ec2_client()
    state = client.describe_instances(InstanceIds=[config.EC2_INSTANCE_ID])[
        "Reservations"
    ][0]["Instances"][0]["State"]["Name"]

    if state == "running":
        print("[ec2] instance already running.")
    else:
        print(f"[ec2] instance state={state}, starting {config.EC2_INSTANCE_ID}...")
        client.start_instances(InstanceIds=[config.EC2_INSTANCE_ID])
        client.get_waiter("instance_running").wait(InstanceIds=[config.EC2_INSTANCE_ID])
        print(f"[ec2] instance running, waiting {config.EC2_BOOT_WAIT_SECONDS}s for the model server to boot...")
        time.sleep(config.EC2_BOOT_WAIT_SECONDS)

    wait_for_handwritten_endpoint_ready()


def stop_ec2_instance():
    if not config.AUTO_MANAGE_EC2:
        return
    if not config.EC2_INSTANCE_ID:
        return
    client = _ec2_client()
    print(f"[ec2] stopping {config.EC2_INSTANCE_ID}...")
    client.stop_instances(InstanceIds=[config.EC2_INSTANCE_ID])


def wait_for_handwritten_endpoint_ready():
    """Polls the vLLM endpoint's /health (or /v1/models) until it responds,
    so main.py doesn't start sending OCR requests before the model server is
    actually up inside the freshly-started instance."""
    import requests

    deadline = time.monotonic() + config.EC2_READY_TIMEOUT_SECONDS
    url = f"{config.HANDWRITTEN_ENDPOINT.rstrip('/')}/health"
    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print("[ec2] handwritten model endpoint is ready.")
                return
        except requests.RequestException:
            pass
        time.sleep(config.EC2_READY_POLL_SECONDS)
    raise TimeoutError(
        f"Handwritten model endpoint at {config.HANDWRITTEN_ENDPOINT} did not become ready "
        f"within {config.EC2_READY_TIMEOUT_SECONDS}s."
    )
