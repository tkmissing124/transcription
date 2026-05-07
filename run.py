"""学び集約システム エントリーポイント

使い方:
    $ uv run python run.py

    タイトル: DX推進セミナー
    日付（YYYY-MM-DD、Enterで今日）: 2026-05-08
    カテゴリ（テクノロジー/ビジネス/読書/セミナー/その他）: セミナー
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

VALID_CATEGORIES = ["テクノロジー", "ビジネス", "読書", "セミナー", "その他"]


# === 対話式入力 ===
def ask_inputs() -> tuple[str, date, str, list[str], str]:
    """タイトル・日付・カテゴリ・発言者・補足を対話式で取得する"""
    print()

    title = ""
    while not title.strip():
        title = input("タイトル: ").strip()
        if not title:
            print("  タイトルは必須です。")

    target_date = None
    while target_date is None:
        raw = input(f"日付（YYYY-MM-DD、Enterで今日 {date.today().isoformat()}）: ").strip()
        if not raw:
            target_date = date.today()
        else:
            try:
                target_date = date.fromisoformat(raw)
            except ValueError:
                print("  形式が正しくありません。YYYY-MM-DD で入力してください。")

    print(f"カテゴリ（{'/'.join(VALID_CATEGORIES)}）")
    category = ""
    while category not in VALID_CATEGORIES:
        category = input("カテゴリ: ").strip()
        if category not in VALID_CATEGORIES:
            print(f"  いずれかを入力してください: {', '.join(VALID_CATEGORIES)}")

    raw_speakers = input("発言者（複数の場合はカンマ区切り、任意）: ").strip()
    speakers = [s.strip() for s in raw_speakers.split(",") if s.strip()] if raw_speakers else []

    context = input("補足（任意、Enterでスキップ）: ").strip()

    print()
    return title, target_date, category, speakers, context


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
def process_one(file_path: Path) -> dict | None:
    kind = classify(file_path)

    if kind == "audio":
        txt = audio.transcribe(file_path)
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


# === 要約Markdown生成 ===
def _build_summary_md(structured: dict, target_date: date, context: str = "") -> str:
    d = target_date.isoformat()
    lines = [
        f"# {d} {structured.get('title', '')}",
        f"カテゴリ: {structured.get('category', '')}",
        "",
        "## 📋 要約",
        structured.get("summary", ""),
        "",
        "## 💡 キーインサイト",
    ]
    for ins in structured.get("key_insights", []) or []:
        lines.append(f"- {ins}")

    lines += ["", "## ✅ アクションアイテム"]
    for item in structured.get("action_items", []) or []:
        lines.append(f"- [ ] {item}")

    if structured.get("highlights"):
        lines += ["", "## 📝 重要抜粋", structured["highlights"]]

    lines += ["", "## 🏷️ キーワード", ", ".join(structured.get("keywords", []) or [])]

    if context:
        lines += ["", "## 補足コンテキスト", context]

    return "\n".join(lines)


# === メイン ===
def main() -> int:
    title, target_date, category, speakers, context = ask_inputs()

    logger.info("=" * 60)
    logger.info("学び集約システム 起動")
    logger.info(f"タイトル: {title} / 日付: {target_date} / カテゴリ: {category}")
    if speakers:
        logger.info(f"発言者: {', '.join(speakers)}")
    logger.info("=" * 60)

    files = collect_files()
    if not files:
        print("inbox/ に処理対象ファイルがありません。")
        logger.info("処理対象ファイルなし。終了します。")
        return 0

    logger.info(f"処理対象: {len(files)}件")
    for f in files:
        logger.info(f"  - {f.name}")

    sources: list[dict] = []
    transcripts: dict[str, str] = {}
    ocrs: dict[str, str] = {}
    source_types: list[str] = []
    success_paths: list[Path] = []
    failed: list[tuple[Path, Exception]] = []

    for f in files:
        try:
            result = process_one(f)
            if result is None:
                failed.append((f, RuntimeError("skipped (unsupported)")))
                continue
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
            failed.append((f, e))

    if not sources:
        logger.warning("処理可能なコンテンツがありません。終了します。")
        for f, _e in failed:
            _move(f, config.ERROR_DIR)
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
        for f in success_paths:
            _move(f, config.ERROR_DIR)
        return 1

    drive_url = ""
    try:
        summary_md = _build_summary_md(structured, target_date=target_date, context=context)
        drive_url = drive_uploader.upload_day(
            target_date=target_date,
            originals=success_paths,
            transcripts=transcripts,
            ocrs=ocrs,
            summary_md=summary_md,
        )
    except Exception as e:
        logger.error(f"Driveアップロード失敗（続行）: {e}")
        logger.error(traceback.format_exc())

    try:
        notion_url = notion_writer.create_page(
            structured=structured,
            source_types=source_types,
            drive_url=drive_url,
            source_count=len(sources),
            processed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            target_date=target_date,
            context=context,
            filenames=[f.name for f in success_paths],
            speakers=speakers,
        )
        print(f"\nNotionページ: {notion_url}")
    except Exception as e:
        logger.error(f"Notion書き込み失敗: {e}")
        logger.error(traceback.format_exc())
        for f in success_paths:
            _move(f, config.ERROR_DIR)
        return 1

    for f in success_paths:
        _move(f, config.DONE_DIR)
    for f, _e in failed:
        _move(f, config.ERROR_DIR)

    print(f"Drive: {drive_url}")
    logger.info(f"完了: 成功 {len(success_paths)}件 / 失敗 {len(failed)}件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
