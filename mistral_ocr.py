import base64
import time
from io import BytesIO

from PIL import Image
import requests

from config import OPENROUTER_API_KEY


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = r"C:\ocr\HC33168_IMG42379_F1.jpg"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# This is the LLM that receives the result produced by Mistral OCR.
# Mistral OCR itself is selected below as the PDF processing engine.
DOWNSTREAM_MODEL = "mistralai/mistral-small-3.2-24b-instruct"

MAX_IMAGE_BYTES = 4_500_000

MAX_RETRIES = 3


# ============================================================
# 1. LOAD AND PREPROCESS IMAGE
# ============================================================

print(f"Loading image: {IMAGE_PATH}")

image = Image.open(IMAGE_PATH).convert("RGB")

# Limit dimensions while preserving aspect ratio.
image.thumbnail((2000, 2000))

print(f"Image size after resize: {image.size}")


# ============================================================
# 2. COMPRESS JPEG
# ============================================================

jpeg_buffer = BytesIO()

quality = 85

while True:
    jpeg_buffer.seek(0)
    jpeg_buffer.truncate(0)

    image.save(
        jpeg_buffer,
        format="JPEG",
        quality=quality,
        optimize=True
    )

    current_size = jpeg_buffer.tell()

    if current_size <= MAX_IMAGE_BYTES or quality <= 40:
        break

    quality -= 5


jpeg_bytes = jpeg_buffer.getvalue()

print(f"JPEG size: {len(jpeg_bytes) / 1024:.1f} KB")
print(f"JPEG quality: {quality}")


# ============================================================
# 3. CONVERT IMAGE TO PDF
#
# OpenRouter's Mistral OCR integration is exposed through
# its PDF parser. Therefore we convert the image to a
# single-page PDF.
# ============================================================

pdf_buffer = BytesIO()

image.save(
    pdf_buffer,
    format="PDF",
    resolution=150.0
)

pdf_bytes = pdf_buffer.getvalue()

print(f"PDF size: {len(pdf_bytes) / 1024:.1f} KB")


# ============================================================
# 4. BASE64 ENCODE PDF
# ============================================================

pdf_base64 = base64.b64encode(pdf_bytes).decode("ascii")

pdf_data_url = f"data:application/pdf;base64,{pdf_base64}"


# ============================================================
# 5. OPENROUTER REQUEST
#
# IMPORTANT:
#
# "mistral-ocr" is NOT the model here.
#
# It is the PDF processing engine.
# ============================================================

payload = {
    "model": DOWNSTREAM_MODEL,

    "plugins": [
        {
            "id": "file-parser",
            "pdf": {
                "engine": "mistral-ocr"
            }
        }
    ],

    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Extract ALL text from this medical document.\n\n"
                        "IMPORTANT:\n"
                        "- Perform transcription only.\n"
                        "- Do not diagnose the patient.\n"
                        "- Do not interpret medical findings.\n"
                        "- Do not summarize.\n"
                        "- Do not correct apparent spelling mistakes.\n"
                        "- Preserve medication names, numbers, doses, units, "
                        "dates, abbreviations and measurements exactly as "
                        "they appear whenever possible.\n"
                        "- If handwriting is unclear, mark it as [UNCLEAR] "
                        "rather than guessing.\n"
                        "- Return the transcription in reading order.\n"
                        "- Return only the extracted/transcribed text."
                    )
                },
                {
                    "type": "file",
                    "file": {
                        "filename": "medical_document.pdf",
                        "file_data": pdf_data_url
                    }
                }
            ]
        }
    ]
}


# ============================================================
# 6. HEADERS
# ============================================================

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}


# ============================================================
# 7. SEND REQUEST WITH RETRIES
# ============================================================

for attempt in range(1, MAX_RETRIES + 1):

    try:

        print(
            f"\nSending request to OpenRouter "
            f"(attempt {attempt}/{MAX_RETRIES})..."
        )

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=180,
        )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.ok:

            result = response.json()

            # ------------------------------------------------
            # EXTRACT RESPONSE TEXT
            # ------------------------------------------------

            content = (
                result
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )

            print("\n" + "=" * 70)
            print("EXTRACTED MEDICAL TEXT")
            print("=" * 70)

            if content:
                print(content)
            else:
                print("No text returned.")

            # ------------------------------------------------
            # USAGE INFORMATION
            # ------------------------------------------------

            usage = result.get("usage")

            if usage:
                print("\n" + "=" * 70)
                print("LLM USAGE")
                print("=" * 70)

                print(f"Prompt tokens:     {usage.get('prompt_tokens')}")
                print(f"Completion tokens: {usage.get('completion_tokens')}")
                print(f"Total tokens:      {usage.get('total_tokens')}")

                if "cost" in usage:
                    print(f"Reported cost:     ${usage['cost']}")

            print("\nModel used:")
            print(result.get("model"))

            break


        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        print(
            f"Attempt {attempt}: "
            f"HTTP {response.status_code}"
        )

        print(response.text)


        # Retry only temporary server/rate-limit errors.
        if response.status_code not in (
            429,
            502,
            503,
            504,
        ):
            response.raise_for_status()


    except requests.RequestException as e:

        print(
            f"Attempt {attempt}: "
            f"Request error: {e}"
        )


    # --------------------------------------------------------
    # RETRY DELAY
    # --------------------------------------------------------

    if attempt < MAX_RETRIES:

        delay = 2 ** (attempt - 1)

        print(f"Retrying in {delay} seconds...")

        time.sleep(delay)

else:

    raise RuntimeError(
        "OpenRouter remained unavailable after "
        f"{MAX_RETRIES} attempts."
    )