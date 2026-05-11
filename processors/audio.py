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


def transcribe(file_path: Path, diarize: bool = False, num_speakers: int | None = None) -> str:
    """音声ファイルをmlx-whisperでローカル文字起こしして返す。"""
    import mlx_whisper

    logger.info(f"音声ファイルを文字起こし中: {file_path.name}")

    result = mlx_whisper.transcribe(
        str(file_path),
        path_or_hf_repo=config.MODEL_ID,
        language="japanese",
        initial_prompt=config.INITIAL_PROMPT or None,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
        verbose=False,
    )

    segments = result.get("segments", [])

    if diarize and segments:
        diarization = _diarize(file_path, num_speakers=num_speakers)
        if diarization:
            segments = _assign_speakers(segments, diarization)
            text = _join_segments_with_speakers(segments)
        else:
            text = _join_segments(segments)
    elif segments:
        text = _join_segments(segments)
    else:
        text = result["text"].strip()

    text = _remove_hallucinations(text)
    logger.info(f"文字起こし完了: {len(text)}文字")
    return text


def _diarize(file_path: Path, num_speakers: int | None = None) -> list[tuple[float, float, str]]:
    """pyannote.audioで話者分離を実行し、[(start, end, speaker_label), ...]を返す"""
    if not config.HF_TOKEN:
        logger.warning("HF_TOKENが未設定のため話者分離をスキップします")
        return []

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        logger.error("pyannote.audioがインストールされていません: uv add pyannote.audio")
        return []

    logger.info("話者分離中（pyannote/speaker-diarization-3.1）...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=config.HF_TOKEN,
    )

    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        logger.info("話者分離: MPSデバイス使用")
    else:
        logger.info("話者分離: CPUデバイス使用")

    kwargs: dict = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    # torchcodec/FFmpegの互換性問題を回避するためtorchaudioで事前ロード
    audio_input = _load_audio_tensor(file_path)
    output = pipeline(audio_input, **kwargs)

    # pyannote のバージョンによって戻り値が異なる (Annotation or DiarizeOutput)
    annotation = _unwrap_annotation(output)

    result = [
        (segment.start, segment.end, speaker)
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]
    logger.info(f"話者分離完了: {len(set(s for _, _, s in result))}名検出")
    return result


def _unwrap_annotation(output):
    """pyannoteの戻り値から Annotation オブジェクトを取り出す。
    バージョンによって Annotation 直接 or DiarizeOutput(dataclass/namedtuple)が返る。"""
    if hasattr(output, "itertracks"):
        return output

    # dataclass の各フィールドを探索
    try:
        import dataclasses
        for field in dataclasses.fields(output):
            val = getattr(output, field.name)
            if hasattr(val, "itertracks"):
                return val
    except TypeError:
        pass

    # NamedTuple / iterable として探索
    try:
        for val in output:
            if hasattr(val, "itertracks"):
                return val
    except TypeError:
        pass

    raise RuntimeError(
        f"Annotation を取り出せません: type={type(output).__name__}, "
        f"attrs={[a for a in dir(output) if not a.startswith('_')]}"
    )


def _load_audio_tensor(file_path: Path) -> dict:
    """ffmpeg CLIでwavに変換してpyannote用テンソルとして返す（torchcodec依存を回避）"""
    import subprocess
    import tempfile
    import numpy as np
    import torch
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(file_path), "-ar", "16000", "-ac", "1", str(tmp_path)],
            capture_output=True,
            check=True,
        )
        waveform_np, sample_rate = sf.read(str(tmp_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform_np.T)  # (channels, time)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"waveform": waveform, "sample_rate": sample_rate}


def _assign_speakers(segments: list[dict], diarization: list[tuple[float, float, str]]) -> list[dict]:
    """各whisperセグメントに最も重複する話者ラベルを付与する"""
    enriched = []
    for seg in segments:
        seg_start, seg_end = seg["start"], seg["end"]
        best_speaker = None
        best_overlap = 0.0

        for d_start, d_end, speaker in diarization:
            overlap = max(0.0, min(seg_end, d_end) - max(seg_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        enriched.append({**seg, "speaker": best_speaker})
    return enriched


def _join_segments_with_speakers(segments: list[dict]) -> str:
    """話者ラベル付きセグメントをテキストに結合する。
    話者が変わったとき、または段落ギャップがあるときに改行する。"""
    lines: list[str] = []
    current_speaker: str | None = None
    current_texts: list[str] = []
    prev_end: float = 0.0

    def _flush(speaker: str | None, texts: list[str]) -> None:
        if not texts:
            return
        label = f"[{speaker}] " if speaker else ""
        lines.append(label + "".join(texts))

    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue

        speaker = seg.get("speaker")
        gap = seg["start"] - prev_end if i > 0 else 0.0

        if speaker != current_speaker:
            _flush(current_speaker, current_texts)
            current_texts = [text]
            current_speaker = speaker
        else:
            if gap >= _PARAGRAPH_GAP:
                _flush(current_speaker, current_texts)
                lines.append("")  # 段落区切り
                current_texts = [text]
            elif gap >= _LINE_GAP:
                _flush(current_speaker, current_texts)
                current_texts = [text]
            else:
                current_texts.append(text)

        prev_end = seg["end"]

    _flush(current_speaker, current_texts)
    return "\n".join(lines)


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
