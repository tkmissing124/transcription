"""OpenAI GPT-4oによる学び内容の構造化"""
import json
import logging

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """あなたは学びを記録・整理するアシスタントです。
ユーザーから渡される複数のソース（音声の文字起こし、手書きメモの読み取り、直接入力のテキスト）を統合し、
以下のJSON形式で構造化してください。

出力は必ず以下のJSON形式のみにしてください。JSON以外のテキストは一切含めないでください。

{
  "summary": "学んだことの要約（3〜6行）",
  "key_insights": ["インサイト1", "インサイト2", "インサイト3"],
  "action_items": ["アクション1", "アクション2"],
  "keywords": ["タグ1", "タグ2", "タグ3"],
  "highlights": "原文の印象的な部分の引用（任意、なければ空文字）"
}

注意事項:
- 原文にない情報を追加・補足しないこと
- 一般論を追加しないこと
- 原文の言い回しやニュアンスを大切にすること
- アクションアイテムが特になければ空配列にすること
- 補足コンテキスト（ある場合）は最優先で反映すること"""


def summarize(sources: list[dict], title: str, category: str, context: str = "") -> dict:
    """
    複数ソースをOpenAI GPT-4oで統合・構造化する。

    Args:
        sources: [{"filename": str, "type": str, "text": str}, ...]
        title: ユーザー指定のタイトル（そのまま使用）
        category: ユーザー指定のカテゴリ（そのまま使用）
        context: ユーザー指定の補足情報（任意）

    Returns:
        構造化された辞書
    """
    logger.info(f"GPT-4oで構造化中: {len(sources)}ソース")
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    parts = []
    if context:
        parts.append(f"## 補足コンテキスト（最優先で反映）\n{context}")

    parts.append("## 素材")
    for i, s in enumerate(sources, 1):
        parts.append(f"### ソース{i}: {s['filename']} (種類: {s['type']})\n{s['text']}")

    parts.append("\n上記を統合して指定JSON形式で出力してください。")
    user_message = "\n\n".join(parts)

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    response_text = response.choices[0].message.content.strip()

    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    result = json.loads(response_text)

    required_keys = {"summary", "key_insights", "action_items", "keywords"}
    missing = required_keys - set(result.keys())
    if missing:
        raise ValueError(f"GPT-4o応答に必須フィールドが不足: {missing}")

    result.setdefault("highlights", "")
    # タイトルとカテゴリはユーザー指定値を使用
    result["title"] = title
    result["category"] = category

    logger.info(f"構造化完了: {result['title']}")
    return result
