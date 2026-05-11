"""学び集約システム エントリーポイント

使い方:
    $ uv run python run.py

    タイトル [26508_会議]:          ← Enterで提案値を使用、入力で上書き
    日付 [2026-05-08]:             ← ファイル名先頭の yymdd から自動推測
    カテゴリ候補: PN、全体、…
    カテゴリ（自由入力可）: セミナー
    発言者（複数の場合はカンマ区切り、任意）: 石川さん, 田中さん
    補足（任意、Enterでスキップ）: 石川さんの話を踏まえて整理

動作:
    1. inbox/ のファイルを収集
    2. 各ファイルを種類別に処理（音声→kotoba-whisper / 画像→GPT-4o OCR / テキスト→読み込み）
    3. OpenAI GPT-4oで統合・構造化
    4. Google Driveに原本アーカイブ
    5. Notionに学びページ作成
    6. 処理済みファイルは done/ に移動（失敗時は error/ に）
"""
import logging
import shutil
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

import config
from processors import audio, image, text
from ai import summarize
import notion_writer
import drive_uploader

# === ログ設定 ===
config.ensure_dirs()
log_file = config.LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run")


# === ファイル名からのデフォルト推測 ===
def _parse_date_from_filename(name: str) -> date | None:
    """
    ファイル名先頭の連続数字から日付を推測する。
    yymdd（5桁）または yymmdd（6桁）に対応。
    例: 26508_xxx → 2026-05-08 / 260508_xxx → 2026-05-08
    """
    digits = ""
    for c in Path(name).stem:
        if c.isdigit():
            digits += c
        else:
            break

    # 6桁: yymmdd
    if len(digits) >= 6:
        try:
            yy, mm, dd = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
            return date(2000 + yy, mm, dd)
        except ValueError:
            pass

    # 5桁: yymdd（月1桁）
    if len(digits) >= 5:
        try:
            yy, m, dd = int(digits[:2]), int(digits[2]), int(digits[3:5])
            return date(2000 + yy, m, dd)
        except ValueError:
            pass

    return None


def _suggest_from_files(files: list[Path]) -> tuple[str, date | None]:
    """inboxの音声ファイルからタイトルと日付のデフォルト値を推測する"""
    audio_files = [f for f in files if f.suffix.lower() in config.AUDIO_EXTENSIONS]
    if not audio_files:
        return "", None
    first = audio_files[0]
    return first.stem, _parse_date_from_filename(first.name)


def _prompt_with_default(prompt: str, default: str) -> str:
    """デフォルト値を表示してinputを受け取る。Enterでデフォルト値を返す。"""
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


# === 対話式入力 ===
def ask_inputs(suggested_title: str = "", suggested_date: date | None = None) -> tuple[str, date, str, list[str], str, bool, bool]:
    """タイトル・日付・カテゴリ・発言者・補足を対話式で取得する"""
    print()

    # タイトル（音声ファイル名をデフォルト提案）
    if suggested_title:
        title = _prompt_with_default("タイトル", suggested_title)
        while not title:
            title = input("タイトル（必須）: ").strip()
    else:
        title = ""
        while not title:
            title = input("タイトル: ").strip()
            if not title:
                print("  タイトルは必須です。")

    # 日付（ファイル名先頭の yymdd から推測）
    default_date = suggested_date or date.today()
    target_date = None
    while target_date is None:
        raw = input(f"日付（YYYY-MM-DD） [{default_date.isoformat()}]: ").strip()
        if not raw:
            target_date = default_date
        else:
            try:
                target_date = date.fromisoformat(raw)
            except ValueError:
                print("  形式が正しくありません。YYYY-MM-DD で入力してください。")

    suggestions = "、".join(config.CATEGORY_SUGGESTIONS)
    print(f"カテゴリ候補: {suggestions}")
    category = ""
    while not category:
        category = input("カテゴリ（自由入力可）: ").strip()
        if not category:
            print("  カテゴリは必須です。")

    raw_speakers = input("発言者（複数の場合はカンマ区切り、任意）: ").strip()
    speakers = [s.strip() for s in raw_speakers.split(",") if s.strip()] if raw_speakers else []

    context = input("補足（任意、Enterでスキップ）: ").strip()

    ocr_images = input("画像をテキスト化して要約に含める？（y/N）: ").strip().lower() == "y"

    diarize = input("話者分離を実行する？（y/N）: ").strip().lower() == "y"

    print()
    return title, target_date, category, speakers, context, ocr_images, diarize


# === ファイル分類 ===
def classify(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in config.AUDIO_EXTENSIONS:
        return "audio"
    if suffix in config.IMAGE_EXTENSIONS:
        return "image"
    if suffix in config.TEXT_EXTENSIONS:
        return "text"
    return "unknown"


def collect_files() -> list[Path]:
    """inbox直下の未処理ファイルを取得"""
    files: list[Path] = []
    for p in sorted(config.INBOX_DIR.iterdir()):
        if p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in config.ALL_EXTENSIONS:
            logger.info(f"対応外の拡張子のためスキップ: {p.name}")
            continue
        files.append(p)
    return files


# === ファイル個別処理 ===
def process_one(file_path: Path, diarize: bool = False, num_speakers: int | None = None) -> dict | None:
    kind = classify(file_path)

    if kind == "audio":
        txt = audio.transcribe(file_path, diarize=diarize, num_speakers=num_speakers)
        return {"filename": file_path.name, "type": "音声", "text": txt, "kind": "audio"}

    if kind == "image":
        txt = image.extract_text(file_path)
        return {"filename": file_path.name, "type": "画像", "text": txt, "kind": "image"}

    if kind == "text":
        txt = text.read_text(file_path)
        return {"filename": file_path.name, "type": "テキスト", "text": txt, "kind": "text"}

    logger.warning(f"種類不明のためスキップ: {file_path.name}")
    return None


# === ファイル移動 ===
def _move(src: Path, dst_dir: Path) -> None:
    if not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        stem, suf = src.stem, src.suffix
        dst = dst_dir / f"{stem}_{datetime.now().strftime('%H%M%S')}{suf}"
    shutil.move(str(src), str(dst))
    logger.info(f"移動: {src.name} -> {dst_dir.name}/")


# === メイン ===
def main() -> int:
    files = collect_files()
    if not files:
        print("inbox/ に処理対象ファイルがありません。")
        return 0

    suggested_title, suggested_date = _suggest_from_files(files)
    title, target_date, category, speakers, context, ocr_images, diarize = ask_inputs(suggested_title, suggested_date)

    # 発言者が指定されている場合、人数をpyannoteに渡す＆コンテキストに追加
    num_speakers = len(speakers) if speakers else None
    if speakers:
        speakers_line = f"発言者: {', '.join(speakers)}"
        context = f"{speakers_line}\n{context}" if context else speakers_line

    logger.info("=" * 60)
    logger.info("学び集約システム 起動")
    logger.info(f"タイトル: {title} / 日付: {target_date} / カテゴリ: {category}")
    if speakers:
        logger.info(f"発言者: {', '.join(speakers)}")
    if diarize:
        logger.info(f"話者分離: 有効（推定人数: {num_speakers or '自動'}）")
    logger.info("=" * 60)

    logger.info(f"処理対象: {len(files)}件")
    for f in files:
        logger.info(f"  - {f.name}")

    sources: list[dict] = []
    transcripts: dict[str, str] = {}
    ocrs: dict[str, str] = {}
    source_types: list[str] = []
    success_paths: list[Path] = []

    for f in files:
        # 画像テキスト化オフの場合はDriveアップロード対象に含めるがOCR・要約はスキップ
        if classify(f) == "image" and not ocr_images:
            logger.info(f"画像テキスト化スキップ（Driveのみ保存）: {f.name}")
            success_paths.append(f)
            continue

        try:
            result = process_one(f, diarize=diarize, num_speakers=num_speakers)
            if result is None:
                logger.error(f"対応外ファイルのため中断: {f.name}")
                return 1
            sources.append({"filename": result["filename"], "type": result["type"], "text": result["text"]})
            source_types.append(result["type"])
            if result["kind"] == "audio":
                transcripts[f.name] = result["text"]
            elif result["kind"] == "image":
                ocrs[f.name] = result["text"]
            success_paths.append(f)
        except Exception as e:
            logger.error(f"ファイル処理失敗: {f.name}: {e}")
            logger.error(traceback.format_exc())
            logger.error("全ファイル処理完了前にエラーが発生したため中断します。")
            return 1

    if not sources:
        logger.warning("処理可能なコンテンツがありません。終了します。")
        return 1

    try:
        structured = summarize.summarize(
            sources=sources,
            title=title,
            category=category,
            context=context,
        )
    except Exception as e:
        logger.error(f"GPT-4o構造化に失敗: {e}")
        logger.error(traceback.format_exc())
        return 1

    try:
        drive_url = drive_uploader.upload_day(
            target_date=target_date,
            originals=success_paths,
            transcripts=transcripts,
            ocrs=ocrs,
            title=title,
        )
    except Exception as e:
        logger.error(f"Driveアップロード失敗: {e}")
        logger.error(traceback.format_exc())
        return 1

    try:
        notion_url = notion_writer.create_page(
            structured=structured,
            drive_url=drive_url,
            source_count=len(sources),
            processed_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            target_date=target_date,
            context=context,
            filenames=[f.name for f in success_paths],
            speakers=speakers,
        )
        print(f"\nNotionページ: {notion_url}")
    except Exception as e:
        logger.error(f"Notion書き込み失敗: {e}")
        logger.error(traceback.format_exc())
        return 1

    done_dir = config.DONE_DIR / target_date.isoformat()
    for f in success_paths:
        _move(f, done_dir)

    print(f"Drive: {drive_url}")
    logger.info(f"完了: 成功 {len(success_paths)}件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
