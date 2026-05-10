"""Notion APIへの学び記録書き込み"""
import logging
from datetime import date

from notion_client import Client

import config

logger = logging.getLogger(__name__)


def _text_block(block_type: str, content: str) -> dict:
    return {
        "object": "block",
        "type": block_type,
        block_type: {
            "rich_text": [{"type": "text", "text": {"content": content}}]
        },
    }


def _heading(level: int, content: str) -> dict:
    bt = f"heading_{level}"
    return _text_block(bt, content)


def _paragraphs(content: str) -> list[dict]:
    blocks = []
    for i in range(0, max(len(content), 1), 1900):
        blocks.append(_text_block("paragraph", content[i : i + 1900] or ""))
    return blocks


def _bookmark(url: str) -> dict:
    return {"object": "block", "type": "bookmark", "bookmark": {"url": url}}


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def build_children(
    structured: dict,
    drive_url: str,
    source_count: int,
    processed_at: str,
    context: str = "",
    filenames: list[str] | None = None,
) -> list[dict]:
    children: list[dict] = []

    # 要約
    children.append(_heading(2, "📋 要約"))
    children.extend(_paragraphs(structured.get("summary", "")))

    # 石川さんの考え方
    ishikawa = structured.get("ishikawa_philosophy", "")
    if ishikawa:
        children.append(_heading(2, "🧠 石川さんの考え方"))
        children.extend(_paragraphs(ishikawa))

    # 重要抜粋
    highlights = structured.get("highlights", "")
    if highlights:
        children.append(_heading(2, "📝 重要抜粋"))
        children.extend(_paragraphs(highlights))

    # 原本リンク
    children.append(_heading(2, "🔗 原本リンク"))
    if drive_url:
        children.append(_bookmark(drive_url))
    else:
        children.extend(_paragraphs("（Drive未連携）"))

    # フッター
    children.append(_divider())
    footer_lines = [f"処理日時: {processed_at}  /  ソース数: {source_count}件"]
    if context:
        footer_lines.append(f"補足: {context}")
    if filenames:
        footer_lines.append(f"ファイル: {', '.join(filenames)}")
    children.extend(_paragraphs("\n".join(footer_lines)))

    return children


def create_page(
    structured: dict,
    source_types: list[str],
    drive_url: str,
    source_count: int,
    processed_at: str,
    target_date: date | None = None,
    context: str = "",
    filenames: list[str] | None = None,
    speakers: list[str] | None = None,
) -> str:
    """
    Notion学びDBにページを作成する。

    Returns:
        作成したページのURL
    """
    logger.info("Notionページを作成中")
    client = Client(auth=config.NOTION_API_KEY)

    d = (target_date or date.today()).isoformat()
    title = structured.get("title", "学び記録")
    category = structured.get("category", "その他")

    properties = {
        "タイトル": {
            "title": [{"type": "text", "text": {"content": title}}]
        },
        "日付": {"date": {"start": d}},
        "カテゴリ": {"select": {"name": category}},
        "ソース種別": {
            "multi_select": [{"name": s} for s in sorted(set(source_types))]
        },
        "ステータス": {"select": {"name": "未読"}},
    }

    if drive_url:
        properties["Driveリンク"] = {"url": drive_url}

    keywords = structured.get("keywords", []) or []
    if keywords:
        properties["キーワード"] = {
            "multi_select": [{"name": kw} for kw in keywords[:5]]
        }

    if speakers:
        properties["発言者"] = {
            "multi_select": [{"name": s} for s in speakers]
        }

    children = build_children(
        structured=structured,
        drive_url=drive_url,
        source_count=source_count,
        processed_at=processed_at,
        context=context,
        filenames=filenames,
    )

    response = client.pages.create(
        parent={"database_id": config.NOTION_DATABASE_ID},
        properties=properties,
        children=children,
    )

    url = response.get("url", "")
    logger.info(f"Notionページ作成完了: {url}")
    return url
