"""音声ファイルの文字起こし（kotoba-whisper、完全ローカル実行）"""
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline():
    """モデルを遅延ロード（初回のみHuggingFaceからDL、以降はキャッシュ）"""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline

        logger.info(f"kotoba-whisperモデルをロード中: {config.KOTOBA_MODEL_ID}")
        _pipeline = pipeline(
            "automatic-speech-recognition",
            model=config.KOTOBA_MODEL_ID,
            device=config.DEVICE,
            chunk_length_s=30,
            batch_size=8,
        )
        logger.info("モデルロード完了")
    return _pipeline


def transcribe(file_path: Path) -> str:
    """
    音声ファイルをkotoba-whisperでローカル文字起こしする。
    ファイルサイズ制限なし。MPS（M1 Neural Engine）使用。
    """
    logger.info(f"音声ファイルを文字起こし中: {file_path.name}")
    pipe = _get_pipeline()
    result = pipe(
        str(file_path),
        generate_kwargs={
            "language": "japanese",
            "initial_prompt": config.INITIAL_PROMPT,
        },
        return_timestamps=False,
    )
    text = result["text"].strip()
    logger.info(f"文字起こし完了: {len(text)}文字")
    return text
