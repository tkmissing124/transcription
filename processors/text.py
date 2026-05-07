"""テキストファイルの読み込み"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_text(file_path: Path) -> str:
    """テキストファイルを読み込む。複数エンコーディングを順に試す。"""
    logger.info(f"テキストファイル読み込み中: {file_path.name}")

    for encoding in ["utf-8", "utf-8-sig", "shift_jis", "euc-jp", "cp932"]:
        try:
            text = file_path.read_text(encoding=encoding).strip()
            logger.info(f"テキスト読み込み完了: {len(text)}文字 (encoding={encoding})")
            return text
        except (UnicodeDecodeError, LookupError):
            continue

    raise ValueError(f"ファイルのエンコーディングを判定できません: {file_path.name}")
