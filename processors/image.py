"""画像ファイルの文字読み取り（OpenAI GPT-4o Vision）"""
import base64
import io
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
}


def _to_jpeg_bytes(file_path: Path) -> bytes:
    """HEICをJPEGに変換して返す。それ以外はそのまま読み込む。"""
    if file_path.suffix.lower() == ".heic":
        import pillow_heif
        from PIL import Image

        pillow_heif.register_heif_opener()
        img = Image.open(file_path)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        return buf.getvalue()
    return file_path.read_bytes()


def extract_text(file_path: Path) -> str:
    """画像からGPT-4o VisionでOCR（手書きメモ・写真の文字読み取り）"""
    logger.info(f"画像からテキスト抽出中: {file_path.name}")

    suffix = file_path.suffix.lower()
    # HEICはJPEGに変換済みなので常にJPEGとして送信
    mime_type = "image/jpeg" if suffix == ".heic" else MIME_MAP.get(suffix, "image/jpeg")
    image_data = base64.b64encode(_to_jpeg_bytes(file_path)).decode("utf-8")

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
                            "あなたは日本語の手書き文字認識の専門家です。"
                            "以下の点に注意して、画像内のテキストを全て読み取ってください。\n\n"
                            "- 崩し字・連続した筆記体も推測して読み取ること\n"
                            "- 読み取れない文字は「□」で表現し、前後の文脈から類推すること\n"
                            "- 固有名詞（人名・地名・カタカナ語）は特に注意して読み取ること\n"
                            "- 矢印・囲み・下線などの構造も「→」「【】」等で表現すること\n"
                            "- 箇条書き・見出しなど元のレイアウト構造を維持すること\n\n"
                            "読み取ったテキストのみを出力し、説明・コメントは不要です。"
                        ),
                    },
                ],
            }
        ],
    )

    text = response.choices[0].message.content.strip()
    logger.info(f"テキスト抽出完了: {len(text)}文字")
    return text
