"""音声ファイルの文字起こし（mlx-whisper、Apple Silicon GPU）"""
import logging
import re
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# 段落区切りと判断する無音時間（秒）
_PARAGRAPH_GAP = 1.5
# 改行と判断する無音時間（秒）
_LINE_GAP = 0.8


def transcribe(file_path: Path) -> str:
    """音声ファイルをmlx-whisperでローカル文字起こしして返す。"""
    import mlx_whisper

    logger.info(f"音声ファイルを文字起こし中: {file_path.name}")

    result = mlx_whisper.transcribe(
        str(file_path),
        path_or_hf_repo=config.KOTOBA_MODEL_ID,
        language="japanese",
        initial_prompt=config.INITIAL_PROMPT or None,
        verbose=False,
    )

    segments = result.get("segments", [])
    if segments:
        text = _join_segments(segments)
    else:
        text = result["text"].strip()

    text = _remove_hallucinations(text)
    logger.info(f"文字起こし完了: {len(text)}文字")
    return text


def _join_segments(segments: list[dict]) -> str:
    """セグメントをタイムスタンプの間隔に応じて改行・段落区切りで結合する。"""
    lines = []
    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue

        if i == 0:
            lines.append(text)
            continue

        gap = seg["start"] - segments[i - 1]["end"]
        if gap >= _PARAGRAPH_GAP:
            lines.append("")  # 空行（段落区切り）
            lines.append(text)
        elif gap >= _LINE_GAP:
            lines.append(text)
        else:
            # 直前のテキストに続けて結合
            if lines:
                lines[-1] = lines[-1] + text
            else:
                lines.append(text)

    return "\n".join(lines)


def _remove_hallucinations(text: str) -> str:
    """連続する同一フレーズの繰り返し（ハルシネーション）を除去する。"""
    # 読点・句点区切りの繰り返し: 「いや、いや、いや、」→「いや、」
    text = re.sub(r'([぀-ヿ一-鿿々ー]+[、。])\1{2,}', r'\1', text)

    # 単語の連続繰り返し: 「うんうんうんうん」→「うん」
    text = re.sub(r'([぀-ヿ一-鿿々ー]{1,6})\1{3,}', r'\1', text)

    # 「お、」「あ、」などの感嘆詞繰り返し
    text = re.sub(r'([あいうえおはい]、){3,}', lambda m: m.group(0)[:m.end() // 3], text)

    return text
