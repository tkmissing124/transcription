"""設定管理"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

# === パス設定 ===
BASE_DIR = Path(__file__).parent
INBOX_DIR = Path("/Users/higashitakahisa/Desktop/transcript/inbox")
DONE_DIR = INBOX_DIR / "done"
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

# === MLX Whisper 設定（Apple Silicon GPU、完全ローカル実行）===
KOTOBA_MODEL_ID = "mlx-community/whisper-large-v3"
# 頻出固有名詞・専門用語をここにカスタマイズ
INITIAL_PROMPT = "PCDO、AA、PN、石川さん"

# === カテゴリ候補（自由入力も可、ここに追加するだけで候補に表示される）===
CATEGORY_SUGGESTIONS = [
    "PN",
    "全体",
    "個別面談（自分）",
    "個別面談",
]

# === 石川さん固有名詞辞書（文字起こし後補正・要約プロンプト用）===
ISHIKAWA_CUSTOM_DICT = {
    "PCDO": "石川氏が運営する社団法人",
    "AA / ALL ACADEMY": "PCDOの上位コミュニティ",
    "V×L": "Value × Leverage の考え方",
    "ESBI": "E(従業員)S(自営)B(ビジネスオーナー)I(投資家)のフレームワーク",
    "PBL": "Project Based Learning",
}

# === OpenAI設定 ===
OPENAI_MODEL = "gpt-4o"

# === ディレクトリ初期化 ===
def ensure_dirs() -> None:
    for d in [INBOX_DIR, DONE_DIR, LOG_DIR, GOOGLE_CLIENT_SECRETS.parent]:
        d.mkdir(parents=True, exist_ok=True)
