"""
Central configuration for the OCR pipeline.
"""
import os
import torch
from dotenv import load_dotenv

load_dotenv()


def _env_int(name, default):
    val = os.environ.get(name)
    return int(val) if val not in (None, "", "None") else default


def _env_float(name, default):
    val = os.environ.get(name)
    return float(val) if val not in (None, "", "None") else default


def _env_bool(name, default):
    val = os.environ.get(name)
    return val.lower() == "true" if val is not None else default


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILTER_JSON = os.path.join(OUTPUT_DIR, "filter.json")     # CLIP classification result (per image)
PENDING_LINKAGE_JSON = os.path.join(OUTPUT_DIR, "pending_linkage.json")

def _env_path(name, default=None):
    val = os.environ.get(name)
    if not val:
        return default
    replacements = {
        "\r": "\\r",
        "\t": "\\t",
        "\n": "\\n",
        "\b": "\\b",
        "\f": "\\f",
        "\a": "\\a",
        "\v": "\\v",
    }
    for escape_char, raw_chars in replacements.items():
        val = val.replace(escape_char, raw_chars)
    return os.path.normpath(val)


DATABASE_URL = os.environ.get("DATABASE_URL")
DB_POOL_MIN = _env_int("DB_POOL_MIN", 2)
DB_POOL_MAX = _env_int("DB_POOL_MAX", 30)
RESULT_CSV_PATH = _env_path("RESULT_CSV_PATH")
DEFAULT_PRIORITY = _env_int("DEFAULT_PRIORITY", 0)

# ---------------------------------------------------------------------------
# DATA SOURCE (constraint: swappable later -- see data/data_fetcher.py)
# ---------------------------------------------------------------------------
DATA_SOURCE = os.environ.get("DATA_SOURCE", "database")  # "drive" today; add "local", "s3", etc. later
DRIVE_API_KEY = os.environ.get("DRIVE_API_KEY")
DRIVE_ROOT_FOLDER_ID = os.environ.get("DRIVE_ROOT_FOLDER_ID")
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DOWNLOAD_WORKERS = _env_int("DOWNLOAD_WORKERS", 8)

FETCH_LIMIT = _env_int("FETCH_LIMIT", 25)          # None disables the cap -- set FETCH_LIMIT= in .env to unset
MAX_RAW_SCAN = _env_int("MAX_RAW_SCAN", 3000)      # safety cap while scanning folders for FETCH_LIMIT usable images

# ---------------------------------------------------------------------------
# STAGE 1 -- CLIP CLASSIFIER (printed vs handwritten) -> filter.json
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_BATCH_SIZE = 64
TILE_GRID = (3, 2)

CLIP_PROMPTS = {
    "printed": (
        "a photo of a machine-printed medical prescription or report, "
        "digital hospital document, typed form with printed divider lines, "
        "table headers, rows, and printed medication names with pen annotations"
    ),
    "handwritten": (
        "a photo of a mostly handwritten medical note or prescription, cursive or "
        "pen-written text with irregular letters, including handwriting on ruled paper "
        "or a form with only a small printed header"
    ),
    "non_document": (
        "a photograph of a person's face, a hand, an object, a selfie, a landscape, "
        "or any image that is not a paper medical document with readable text"
    ),
}

# -- CLIP-specific thresholds (tile-vote based). Do not reuse these for Paddle/EC2.
HANDWRITTEN_THRESHOLD = _env_float("HANDWRITTEN_THRESHOLD", 0.65)
HANDWRITTEN_MARGIN = _env_float("HANDWRITTEN_MARGIN", 0.20)
PRINTED_SHORTCUT_THRESHOLD = _env_float("PRINTED_SHORTCUT_THRESHOLD", 0.55)
PRINTED_SHORTCUT_MARGIN = _env_float("PRINTED_SHORTCUT_MARGIN", 0.20)
HANDWRITTEN_SHORTCUT_ENABLED = True
HANDWRITTEN_SHORTCUT_THRESHOLD = _env_float("HANDWRITTEN_SHORTCUT_THRESHOLD", 0.95)
HANDWRITTEN_SHORTCUT_MARGIN = _env_float("HANDWRITTEN_SHORTCUT_MARGIN", 0.50)
HANDWRITTEN_TILE_RATIO = _env_float("HANDWRITTEN_TILE_RATIO", 0.75)
MIN_NON_BLANK_TILES_FOR_HANDWRITTEN = _env_int("MIN_NON_BLANK_TILES_FOR_HANDWRITTEN", 4)
MIN_TILE_WEIGHT = _env_float("MIN_TILE_WEIGHT", 0.08)
NON_DOCUMENT_THRESHOLD = _env_float("NON_DOCUMENT_THRESHOLD", 0.45)
NON_DOCUMENT_MARGIN = _env_float("NON_DOCUMENT_MARGIN", 0.08)
DOCUMENT_GATE_VERSION = "3"  # bump when routing logic changes to invalidate filter.json

STRUCTURAL_OVERRIDE_ENABLED = True
STRUCTURAL_MIN_LINES = 4
STRUCTURAL_MIN_LINE_FRAC = 0.35
STRUCTURAL_REQUIRE_BOTH_AXES = True

CLIP_MAX_DIMENSION = 2000  # size cap applied to the source array before tiling for CLIP

# fingerprint so filter.json entries auto-invalidate if you change any threshold above
import hashlib as _hashlib
import json as _json


def classifier_config_fingerprint():
    parts = [
        CLIP_MODEL_NAME, CLIP_PRETRAINED, str(TILE_GRID),
        str(STRUCTURAL_OVERRIDE_ENABLED), str(STRUCTURAL_MIN_LINES), str(STRUCTURAL_MIN_LINE_FRAC),
        str(STRUCTURAL_REQUIRE_BOTH_AXES),
        str(HANDWRITTEN_THRESHOLD), str(HANDWRITTEN_MARGIN),
        str(PRINTED_SHORTCUT_THRESHOLD), str(PRINTED_SHORTCUT_MARGIN),
        str(HANDWRITTEN_SHORTCUT_ENABLED), str(HANDWRITTEN_SHORTCUT_THRESHOLD), str(HANDWRITTEN_SHORTCUT_MARGIN),
        str(HANDWRITTEN_TILE_RATIO), str(MIN_NON_BLANK_TILES_FOR_HANDWRITTEN),
        str(MIN_TILE_WEIGHT), _json.dumps(CLIP_PROMPTS, sort_keys=True),
        str(NON_DOCUMENT_THRESHOLD), str(NON_DOCUMENT_MARGIN),
        DOCUMENT_GATE_VERSION,
    ]
    return _hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


CLASSIFIER_CONFIG_FINGERPRINT = classifier_config_fingerprint()

# ---------------------------------------------------------------------------
# STAGE 2 -- PRINTED MODEL: PaddleOCR (own preprocessing + own thresholds)
# ---------------------------------------------------------------------------
PADDLE_MAX_DIMENSION = 2000  # printed docs are usually high-res scans -> keep detail
PADDLE_CONFIDENCE_THRESHOLD = _env_float("PADDLE_CONFIDENCE_THRESHOLD", 0.5)
PADDLE_LONG_TEXT_MIN_CHARS = 25
PADDLE_LONG_TEXT_CONFIDENCE_THRESHOLD = _env_float("PADDLE_LONG_TEXT_CONFIDENCE_THRESHOLD", 0.35)
PADDLE_BATCH_SIZE = _env_int("PADDLE_BATCH_SIZE", 16)
PADDLE_VOCAB_MATCH_THRESHOLD = 82

PADDLE_DET_MODEL = "PP-OCRv5_server_det"
PADDLE_REC_MODEL_EN = "en_PP-OCRv5_mobile_rec"
PADDLE_REC_MODEL_SERVER = "PP-OCRv5_server_rec"
PADDLE_DET_LIMIT_SIDE_LEN = 1600
PADDLE_DET_THRESH = 0.2
PADDLE_DET_BOX_THRESH = 0.5
PADDLE_DET_UNCLIP_RATIO = 1.6
PADDLE_DEVICE = os.environ.get("PADDLE_DEVICE", "gpu:0")  # constraint: GPU build only, no CPU fallback

ROW_TOLERANCE = 15
COLUMN_GAP_THRESHOLD = 120

LAB_VOCAB = [
    "Investigation", "Result", "Units", "Bio. Ref. Interval", "Interpretation",
    "Biochemistry", "Patient ID", "Patient Name", "Ref. By Doctor", "Facility Name",
    "Age/Sex", "Sample Coll. Date", "Reg. Date/Time", "Report Date/Time",
    "Sample Type: Serum", "Method: CHOD-PAP", "Method: GPO", "Method: Calculated",
    "Method: Immunoinhibition", "End Of Report",
    "Serum Total Cholesterol", "Serum Triglycerides", "Serum VLDL-Cholesterol",
    "Serum HDL-Cholesterol", "Serum LDL-Cholesterol", "Desirable", "Borderline High",
    "High", "Very High", "Normal",
    "Pulse", "Pulse Rate", "Heart Rate", "HR",
    "SpO2", "SPO2", "Oxygen Saturation",
    "Blood Pressure", "BP", "Systolic", "Diastolic", "mmHg",
    "Temperature", "Temp", "RR", "Respiratory Rate", "Resp. Rate",
    "BMI", "Height", "Weight", "bpm", "beats/min",
    "Vitals", "Vital Signs", "General Examination",
]

VITALS_UNIT_HINTS = {
    "bpm": "Pulse", "beats/min": "Pulse",
    "mmhg": "Blood Pressure",
    "%": "SpO2",
    "\u00b0f": "Temperature", "\u00b0c": "Temperature",
}

# ---------------------------------------------------------------------------
# STAGE 3 -- HANDWRITTEN MODEL: EC2-hosted vLLM (own preprocessing, no OCR score gate)
# ---------------------------------------------------------------------------
HANDWRITTEN_MAX_DIMENSION = 720  # much smaller than Paddle's -- vLLM is far more size/token sensitive
HANDWRITTEN_ENDPOINT = os.environ.get("HANDWRITTEN_ENDPOINT", "http://127.0.0.1:8000")
HANDWRITTEN_TIMEOUT = _env_int("HANDWRITTEN_TIMEOUT", 120)
HANDWRITTEN_MAX_RETRIES = _env_int("HANDWRITTEN_MAX_RETRIES", 3)
HANDWRITTEN_CONCURRENCY = _env_int("HANDWRITTEN_CONCURRENCY", 24)
HANDWRITTEN_PROMPT = (
    "Transcribe all text visible in this image exactly as written.\n\n"
    "Do not correct, complete, or infer text.\n"
    "Preserve the original reading order and line breaks.\n"
    "If text cannot be read, write [ILLEGIBLE].\n"
    "Output only the transcription."
)

# EC2 auto start/stop (future_work item: "automatically turning on the ec2 run,
# save in result.json, then turning it off"). If AUTO_MANAGE_EC2=false, main.py
# assumes you already have the SSH tunnel to the endpoint open manually.
AUTO_MANAGE_EC2 = _env_bool("AUTO_MANAGE_EC2", False)
EC2_INSTANCE_ID = os.environ.get("EC2_INSTANCE_ID")
EC2_REGION = os.environ.get("EC2_REGION", "ap-south-1")
EC2_BOOT_WAIT_SECONDS = _env_int("EC2_BOOT_WAIT_SECONDS", 90)
EC2_READY_POLL_SECONDS = _env_int("EC2_READY_POLL_SECONDS", 10)
EC2_READY_TIMEOUT_SECONDS = _env_int("EC2_READY_TIMEOUT_SECONDS", 600)

# ---------------------------------------------------------------------------
# PIPELINE-WIDE
# ---------------------------------------------------------------------------
# SAVE_EVERY = _env_int("SAVE_EVERY", 10)
# Checkpoint frequency -- kept separate per stage because your original
# scripts didn't agree: test.py saved output.json every 50 images and the
# handwriting_filter cache every 200; handwritten.py saved result.json every 10.
PADDLE_SAVE_EVERY = _env_int("PADDLE_SAVE_EVERY", 50)
HANDWRITTEN_SAVE_EVERY = _env_int("HANDWRITTEN_SAVE_EVERY", 10)
CLIP_CACHE_SAVE_EVERY = _env_int("CLIP_CACHE_SAVE_EVERY", 200)
