"""画像ファイルの文字読み取り（OpenAI GPT-4o Vision）"""
import base64
import logging
from pathlib import Path

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".heic": "image/heic",
}


def extract_text(file_path: Path) -> str:
    """画像からGPT-4o VisionでOCR（手書きメモ・写真の文字読み取り）"""
    logger.info(f"画像からテキスト抽出中: {file_path.name}")

    suffix = file_path.suffix.lower()
    mime_type = MIME_MAP.get(suffix, "image/jpeg")
    image_data = base64.b64encode(file_path.read_bytes()).decode("utf-8")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "この画像に含まれるテキストを全て読み取ってください。"
                            "手書きの場合も可能な限り正確に読み取ってください。"
                            "読み取ったテキストのみを出力し、それ以外の説明は不要です。"
                            "メモや手書きノートの場合は、構造（箇条書き・見出しなど）も維持してください。"
                        ),
                    },
                ],
            }
        ],
    )

    text = response.choices[0].message.content.strip()
    logger.info(f"テキスト抽出完了: {len(text)}文字")
    return text
