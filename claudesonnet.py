import base64
import time
from io import BytesIO
from PIL import Image

import requests

from config import OPENROUTER_API_KEY


image_path = r"C:\ocr\HC8972_IMG4496_F4.jpg"
image = Image.open(image_path).convert("RGB")
image.thumbnail((2000, 2000))

buffer = BytesIO()
quality = 85

while True:
    buffer.seek(0)
    buffer.truncate()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)

    if buffer.tell() <= 4_500_000 or quality <= 40:
        break

    quality -= 5

image_data = base64.b64encode(buffer.getvalue()).decode("ascii")

payload = {
    "model": "anthropic/claude-sonnet-4",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all medical text in the document."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    },
                },
            ],
        }
    ],
}

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

for attempt in range(3):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )

    if response.ok:
        print(response.json()["choices"][0]["message"]["content"])
        break

    print(f"Attempt {attempt + 1}: HTTP {response.status_code}")
    print(response.text)

    if response.status_code not in (502, 503, 504):
        response.raise_for_status()

    if attempt < 2:
        time.sleep(2 ** attempt)
else:
    raise RuntimeError("OpenRouter remained unavailable after 3 attempts.")