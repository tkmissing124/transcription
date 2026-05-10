"""Google Drive連携（OAuth 2.0 / ローカルアプリ）"""
import logging
import mimetypes
import tempfile
from datetime import date
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import config

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _get_service():
    """OAuth 2.0で認証済みのDriveサービスを返す。初回はブラウザで認可。"""
    creds: Credentials | None = None
    token_path = config.GOOGLE_TOKEN_FILE

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not config.GOOGLE_CLIENT_SECRETS.exists():
                raise FileNotFoundError(
                    f"Google OAuthクライアントシークレットが見つかりません: "
                    f"{config.GOOGLE_CLIENT_SECRETS}\n"
                    f"GCPコンソールで『デスクトップアプリ』のOAuthクライアントを作成し、"
                    f"JSONを上記パスに配置してください。"
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(config.GOOGLE_CLIENT_SECRETS), SCOPES
            )
            creds = flow.run_local_server(port=0)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _find_or_create_folder(service, name: str, parent_id: str | None) -> str:
    safe_name = name.replace("'", "\\'")
    q = [
        "mimeType = 'application/vnd.google-apps.folder'",
        "trashed = false",
        f"name = '{safe_name}'",
    ]
    if parent_id:
        q.append(f"'{parent_id}' in parents")

    res = service.files().list(
        q=" and ".join(q),
        fields="files(id, name)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    files = res.get("files", [])
    if files:
        return files[0]["id"]

    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    folder = service.files().create(body=meta, fields="id", supportsAllDrives=True).execute()
    return folder["id"]


def _ensure_daily_folder(service, target_date: date) -> tuple[str, str]:
    """YYYY/YYYY-MM/YYYY-MM-DD のフォルダ階層を作成し、(folder_id, folder_url) を返す"""
    root_id = config.GOOGLE_DRIVE_ROOT_FOLDER_ID or None
    root_id = _find_or_create_folder(service, config.GOOGLE_DRIVE_ROOT_FOLDER_NAME, root_id)

    year_id = _find_or_create_folder(service, target_date.strftime("%Y"), root_id)
    month_id = _find_or_create_folder(service, target_date.strftime("%Y-%m"), year_id)
    day_id = _find_or_create_folder(service, target_date.strftime("%Y-%m-%d"), month_id)

    url = f"https://drive.google.com/drive/folders/{day_id}"
    return day_id, url


def _upload_file(service, folder_id: str, local_path: Path, name: str | None = None) -> str:
    mime, _ = mimetypes.guess_type(str(local_path))
    mime = mime or "application/octet-stream"
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=False)
    meta = {"name": name or local_path.name, "parents": [folder_id]}
    file = service.files().create(
        body=meta, media_body=media, fields="id", supportsAllDrives=True
    ).execute()
    return file["id"]


def upload_day(
    target_date: date,
    originals: list[Path],
    transcripts: dict[str, str],
    ocrs: dict[str, str],
    title: str = "",
) -> str:
    """
    指定日のフォルダに原本・文字起こし・OCR・要約を保存する。

    Returns:
        日次フォルダのURL
    """
    logger.info(f"Driveへアップロード開始: {target_date.isoformat()}")
    service = _get_service()
    folder_id, folder_url = _ensure_daily_folder(service, target_date)

    for p in originals:
        try:
            _upload_file(service, folder_id, p)
        except Exception as e:
            logger.warning(f"原本アップロード失敗 {p.name}: {e}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        if transcripts:
            transcript_name = f"{title}_文字起こし.txt" if title else "transcript.txt"
            t_path = tmp_dir / transcript_name
            body = "\n\n".join(f"# {k}\n{v}" for k, v in transcripts.items())
            t_path.write_text(body, encoding="utf-8")
            _upload_file(service, folder_id, t_path, name=transcript_name)

        if ocrs:
            o_path = tmp_dir / "ocr.txt"
            body = "\n\n".join(f"# {k}\n{v}" for k, v in ocrs.items())
            o_path.write_text(body, encoding="utf-8")
            _upload_file(service, folder_id, o_path)

    logger.info(f"Driveアップロード完了: {folder_url}")
    return folder_url
