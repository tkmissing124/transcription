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
  "keywords": ["タグ1", "タグ2", "タグ3"],
  "highlights": "原文の印象的な部分の引用（任意、なければ空文字）",
  "ishikawa_philosophy": "（石川氏の発言が含まれる場合のみ。なければ空文字）"
}

注意事項:
- 原文にない情報を追加・補足しないこと
- 一般論を追加しないこと
- 原文の言い回しやニュアンスを大切にすること
- 補足コンテキスト（ある場合）は最優先で反映すること"""


def _build_ishikawa_dict_str() -> str:
    lines = []
    for term, desc in config.ISHIKAWA_CUSTOM_DICT.items():
        lines.append(f"- {term}: {desc}")
    return "\n".join(lines)


ISHIKAWA_PROMPT = """あなたは石川氏の思考・哲学を抽出する専門家です。
以下の文字起こしから、石川氏の発言のみを対象に抽出してください。

【カスタム辞書（誤変換補正）】
{dict_str}

【出力形式】
### 核となる主張
（石川氏が最も伝えたかったこと）

### 論理構造
（どのような論拠・事例でその主張を展開したか）

### 独自の言葉・定義
（石川氏が使う特有の表現や概念の定義）

【禁止事項】
- AIによる一般論・アドバイスの追加禁止
- 石川氏が言っていない内容の補完禁止
- 石川氏の発言が含まれない・不明な場合は「（石川氏の発言は確認できませんでした）」とだけ出力"""


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

    required_keys = {"summary", "keywords"}
    missing = required_keys - set(result.keys())
    if missing:
        raise ValueError(f"GPT-4o応答に必須フィールドが不足: {missing}")

    result.setdefault("highlights", "")
    result.setdefault("ishikawa_philosophy", "")

    # 石川さんの考え方を別途抽出
    has_audio = any(s["type"] == "音声" for s in sources)
    if has_audio:
        result["ishikawa_philosophy"] = _extract_ishikawa(client, sources)

    # タイトルとカテゴリはユーザー指定値を使用
    result["title"] = title
    result["category"] = category

    logger.info(f"構造化完了: {result['title']}")
    return result


def _extract_ishikawa(client: OpenAI, sources: list[dict]) -> str:
    audio_sources = [s for s in sources if s["type"] == "音声"]
    if not audio_sources:
        return ""

    combined = "\n\n".join(f"# {s['filename']}\n{s['text']}" for s in audio_sources)
    prompt = ISHIKAWA_PROMPT.format(dict_str=_build_ishikawa_dict_str())

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": combined},
        ],
    )
    return response.choices[0].message.content.strip()
