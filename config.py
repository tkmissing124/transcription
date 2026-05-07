"""設定管理"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

# === パス設定 ===
BASE_DIR = Path(__file__).parent
INBOX_DIR = Path("/Users/higashitakahisa/Desktop/transcription/inbox")
DONE_DIR = INBOX_DIR / "done"
ERROR_DIR = INBOX_DIR / "error"
LOG_DIR = BASE_DIR / "logs"

# === APIキー ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")

# === Google Drive設定（OAuth 2.0）===
GOOGLE_CLIENT_SECRETS = BASE_DIR / "credentials" / "client_secret.json"
GOOGLE_TOKEN_FILE = BASE_DIR / "credentials" / "token.json"
GOOGLE_DRIVE_ROOT_FOLDER_NAME = os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_NAME", "学びアーカイブ")
GOOGLE_DRIVE_ROOT_FOLDER_ID = os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_ID", "")

# === 対応ファイル形式 ===
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".webm", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md"}
ALL_EXTENSIONS = AUDIO_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS

# === kotoba-whisper 設定（完全ローカル実行）===
KOTOBA_MODEL_ID = "kotoba-tech/kotoba-whisper-v2.0"
DEVICE = "mps"  # M1 Neural Engine
# 頻出固有名詞・専門用語をここにカスタマイズ
INITIAL_PROMPT = "PCDO、AA、PN、石川さん"

# === OpenAI設定 ===
OPENAI_MODEL = "gpt-4o"

# === ディレクトリ初期化 ===
def ensure_dirs() -> None:
    for d in [INBOX_DIR, DONE_DIR, ERROR_DIR, LOG_DIR, GOOGLE_CLIENT_SECRETS.parent]:
        d.mkdir(parents=True, exist_ok=True)
