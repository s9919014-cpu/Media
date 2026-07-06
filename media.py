"""MediaGrab AI Bot — single-file consolidation.

A production-ready, fully-async Telegram bot specialised in:
  • Downloading every kind of Telegram media.
  • AI-powered analysis using Google Gemini (5 modes).
  • A clean, modern inline-button UI (no commands required) + inline mode.

Architecture: OFFICIAL Telegram Bot API as the primary interface, plus an
OPTIONAL MTProto (Telethon) userbot backend for restricted content /
self-destruct media capture. The MTProto backend is lazy-imported: the bot
runs fine without Telethon installed.

This file consolidates the original modular package (config / utils /
database / services / ui / handlers / bot) into a single runnable module.
Save as ``bot.py`` and run ``python bot.py``.

Run:
    python bot.py            # long-polling (default)
    USE_WEBHOOK=true python bot.py   # webhook mode
"""
from __future__ import annotations
import asyncio, os, re, json, time, signal, sys, shutil, functools, logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypeVar, Sequence, Iterable, Mapping
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from types import SimpleNamespace

import httpx
import aiofiles
import aiosqlite
from dotenv import load_dotenv
from PIL import Image
import qrcode
from telegram import (Update, InlineKeyboardButton, InlineKeyboardMarkup,
                      InlineQueryResultArticle, InputTextMessageContent)
from telegram.ext import (Application, ApplicationBuilder, CallbackQueryHandler,
                          CommandHandler, InlineQueryHandler, MessageHandler,
                          ContextTypes, filters)
from telegram.error import BadRequest, Forbidden, NetworkError, TimedOut, RetryAfter


# ---------------------------------------------------------------------------
# Python 3.12 dataclass + importlib quirk workaround.
# When this file is loaded via ``importlib.util.spec_from_file_location``
# without pre-registering in ``sys.modules``, ``@dataclass(frozen=True)``
# combined with ``from __future__ import annotations`` fails because the
# dataclasses internal ``_is_type()`` does
# ``sys.modules.get(cls.__module__).__dict__`` and gets ``None``. We
# self-register a placeholder module here so that lookup succeeds; the
# placeholder's empty ``__dict__`` makes the ClassVar/InitVar detection
# correctly return False for our normal fields. When run as a script
# (``__name__ == '__main__'``) or imported normally, ``sys.modules`` already
# has the entry and this is a no-op.
# ---------------------------------------------------------------------------
if __name__ not in sys.modules:
    sys.modules[__name__] = type(sys)(__name__)


# ===========================================================================
# Config
# ===========================================================================

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_list_int(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name, "")
    if not raw:
        return list(default)
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out or list(default)


# Official Telegram Bot API hard limits.
#   * Cloud Bot API getFile   : 20 MB
#   * Cloud Bot API send      : 50 MB
#   * Local Bot API server    : 2 GB for both (true hard ceiling).
CLOUD_BOT_API_DOWNLOAD_LIMIT_BYTES = 20 * 1024 * 1024
CLOUD_BOT_API_UPLOAD_LIMIT_BYTES = 50 * 1024 * 1024
LOCAL_BOT_API_LIMIT_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration."""

    # --- Telegram (Bot API only) ---
    bot_token: str = _env("TG_BOT_TOKEN")

    # --- Local Bot API server (optional, for large files) ---
    local_mode: bool = _env_bool("LOCAL_MODE", False)
    bot_api_base_url: str = _env("BOT_API_BASE_URL", "http://localhost:8081")
    bot_api_file_url: str = _env("BOT_API_FILE_URL", "http://localhost:8081")

    # --- AI ---
    gemini_api_key: str = _env("GEMINI_API_KEY")

    # --- Admin ---
    admin_ids: tuple[int, ...] = tuple(_env_list_int("ADMIN_IDS", []))

    # --- Paths ---
    project_root: Path = _PROJECT_ROOT
    db_path: Path = Path(_env("DB_PATH", str(_PROJECT_ROOT / "data" / "bot.db")))
    downloads_dir: Path = Path(_env("DOWNLOADS_DIR", str(_PROJECT_ROOT / "data" / "downloads")))
    frames_dir: Path = Path(_env("FRAMES_DIR", str(_PROJECT_ROOT / "data" / "frames")))

    # --- Limits / performance ---
    max_file_size_mb: int = _env_int("MAX_FILE_SIZE_MB", 5120)
    max_concurrent_downloads: int = _env_int("MAX_CONCURRENT_DOWNLOADS", 4)
    chunk_size_kb: int = _env_int("CHUNK_SIZE_KB", 512)
    download_timeout: int = _env_int("DOWNLOAD_TIMEOUT", 3600)
    max_retries: int = _env_int("MAX_RETRIES", 5)
    num_frames: int = _env_int("NUM_FRAMES", 6)

    # --- Per-user quotas ---
    user_daily_download_limit: int = _env_int("USER_DAILY_DOWNLOAD_LIMIT", 100)
    user_daily_download_bytes_mb: int = _env_int("USER_DAILY_DOWNLOAD_BYTES_MB", 5120)

    # --- Defaults (overridable per-user via Settings) ---
    default_quality: str = "1080p"
    default_ai_model: str = "gemini-2.0-flash"
    default_language: str = "English"

    log_level: str = _env("LOG_LEVEL", "INFO").upper()

    # --- Webhook (optional; falls back to long-polling) ---
    use_webhook: bool = _env_bool("USE_WEBHOOK", False)
    webhook_url: str = _env("WEBHOOK_URL", "")           # public https URL
    webhook_port: int = _env_int("WEBHOOK_PORT", 8443)
    webhook_path: str = _env("WEBHOOK_PATH", "/webhook")
    webhook_listen: str = _env("WEBHOOK_LISTEN", "127.0.0.1")

    # --- Inline mode ---
    inline_enabled: bool = _env_bool("INLINE_ENABLED", True)

    # --- MTProto backend (optional) ---
    mtproto_enabled: bool = _env_bool("MTPROTO_ENABLED", False)
    tg_api_id: int = _env_int("TG_API_ID", 0)
    tg_api_hash: str = _env("TG_API_HASH")
    tg_user_session: str = _env("TG_USER_SESSION", "mediagrab_user")
    mtproto_admin_id: int = _env_int("MTPROTO_ADMIN_ID", 0)

    # --- VC Tour (voice-chat tour, optional) ---
    vc_tour_enabled: bool = _env_bool("VC_TOUR_ENABLED", False)
    vc_stay_minutes: int = _env_int("VC_STAY_MINUTES", 5)
    vc_cooldown_seconds: int = _env_int("VC_COOLDOWN_SECONDS", 30)
    vc_revisit_same_group: bool = _env_bool("VC_REVISIT_SAME_GROUP", False)
    vc_auto_resume_after_manual: bool = _env_bool("VC_AUTO_RESUME_AFTER_MANUAL", True)
    vc_join_notifications: bool = _env_bool("VC_JOIN_NOTIFICATIONS", True)
    vc_leave_notifications: bool = _env_bool("VC_LEAVE_NOTIFICATIONS", True)
    vc_save_history: bool = _env_bool("VC_SAVE_HISTORY", True)
    vc_discovery_limit: int = _env_int("VC_DISCOVERY_LIMIT", 200)
    vc_min_stay_minutes: int = _env_int("VC_MIN_STAY_MINUTES", 1)
    vc_max_stay_minutes: int = _env_int("VC_MAX_STAY_MINUTES", 60)

    @property
    def mtproto_configured(self) -> bool:
        """True if MTProto can be started (enabled + has credentials)."""
        return (self.mtproto_enabled
                and self.tg_api_id
                and self.tg_api_hash
                and self.tg_user_session)

    @property
    def vc_admin_ids(self) -> tuple[int, ...]:
        """IDs authorised to control the VC tour (admin IDs + MTProto admin)."""
        ids = set(self.admin_ids)
        if self.mtproto_admin_id:
            ids.add(self.mtproto_admin_id)
        return tuple(ids)

    # --- Derived ---
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def chunk_size_bytes(self) -> int:
        return max(64 * 1024, self.chunk_size_kb * 1024)

    @property
    def download_limit_bytes(self) -> int:
        """Effective download ceiling: the configured limit (no artificial cap)."""
        return self.max_file_size_bytes

    @property
    def upload_limit_bytes(self) -> int:
        """Bot API sendDocument ceiling. Cloud=50MB, local=2GB."""
        if self.local_mode:
            return LOCAL_BOT_API_LIMIT_BYTES
        return CLOUD_BOT_API_UPLOAD_LIMIT_BYTES

    @property
    def user_daily_download_bytes(self) -> int:
        return self.user_daily_download_bytes_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for path in (self.downloads_dir, self.frames_dir, self.db_path.parent):
            path.mkdir(parents=True, exist_ok=True)


config = Config()
config.ensure_dirs()


# ===========================================================================
# Logger
# ===========================================================================

_LOG_FMT = (
    "%(asctime)s | %(levelname)-7s | %(name)-18s | "
    "%(funcName)s:%(lineno)d | %(message)s"
)
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> logging.Logger:
    """Configure root logging once. Safe to call multiple times."""
    root = logging.getLogger()
    if getattr(root, "_mediagrab_configured", False):
        return logging.getLogger("mediagrab")

    level = getattr(logging, config.log_level, logging.INFO)
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        log_file = config.project_root / "bot.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Read-only filesystem etc. — fall back to console-only.
        pass

    # Silence noisy libraries.
    for noisy in ("PIL", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._mediagrab_configured = True  # type: ignore[attr-defined]
    return logging.getLogger("mediagrab")


logger = setup_logging()


# ===========================================================================
# Utils — filenames
# ===========================================================================

_FORBIDDEN_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_MAX_NAME_LEN = 180


def safe_filename(name: str, fallback: str = "file") -> str:
    """Return a filesystem-safe filename."""
    if not name:
        return fallback
    cleaned = _FORBIDDEN_CHARS.sub("_", name).strip(" .")
    if not cleaned:
        cleaned = fallback
    base, dot, ext = cleaned.rpartition(".")
    if not base:  # no extension / hidden file
        base, ext = cleaned, ""
        dot = ""
    if base.upper() in _RESERVED_NAMES:
        base = f"_{base}"
        cleaned = f"{base}{dot}{ext}" if dot else base
    if len(cleaned) > _MAX_NAME_LEN:
        if ext:
            base = base[: _MAX_NAME_LEN - len(ext) - 1]
            cleaned = f"{base}.{ext}"
        else:
            cleaned = cleaned[:_MAX_NAME_LEN]
    return cleaned


def unique_path(directory, filename: str):
    """Append (1), (2)… before the extension if the file already exists."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if not target.exists():
        return target
    stem, dot, ext = filename.rpartition(".")
    suffix = f".{ext}" if dot else ""
    counter = 1
    while True:
        candidate = directory / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


_EXTENSION_CATEGORIES: dict[str, str] = {}

_VIDEO_EXTS = {"mp4", "mkv", "avi", "mov", "webm", "flv", "wmv", "m4v", "mpg", "mpeg", "ts", "3gp"}
_AUDIO_EXTS = {"mp3", "m4a", "aac", "flac", "ogg", "opus", "wav", "wma"}
_ARCHIVE_EXTS = {"zip", "rar", "7z", "tar", "gz", "bz2", "xz"}
_DOC_EXTS = {"pdf", "epub", "txt", "doc", "docx", "rtf", "mobi", "azw3"}
_APP_EXTS = {"apk", "xapk", "apks", "ipa", "deb", "rpm", "dmg", "exe", "msi"}
_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff", "heic"}
_ISO_EXTS = {"iso", "img", "bin", "vhd", "vmdk"}
_SUBTITLE_EXTS = {"srt", "ass", "ssa", "vtt", "sub"}

for _ext in _VIDEO_EXTS:
    _EXTENSION_CATEGORIES[_ext] = "video"
for _ext in _AUDIO_EXTS:
    _EXTENSION_CATEGORIES[_ext] = "audio"
for _ext in _ARCHIVE_EXTS:
    _EXTENSION_CATEGORIES[_ext] = "archive"
for _ext in _DOC_EXTS:
    _EXTENSION_CATEGORIES[_ext] = "document"
for _ext in _APP_EXTS:
    _EXTENSION_CATEGORIES[_ext] = "app"
for _ext in _IMAGE_EXTS:
    _EXTENSION_CATEGORIES[_ext] = "image"
for _ext in _ISO_EXTS:
    _EXTENSION_CATEGORIES[_ext] = "iso"
for _ext in _SUBTITLE_EXTS:
    _EXTENSION_CATEGORIES[_ext] = "subtitle"


def extension_category(filename: str | None) -> str:
    """Classify a filename by its extension. Falls back to 'document'."""
    if not filename:
        return "document"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _EXTENSION_CATEGORIES.get(ext, "document")


def telegram_media_type(message) -> str | None:
    """Map a python-telegram-bot Message to a media-type string."""
    if message is None:
        return None
    if message.video:
        return "video"
    if message.animation:  # GIF
        return "gif"
    if message.sticker:
        return "sticker"
    if message.video_note:
        return "video_note"
    if message.voice:
        return "voice"
    if message.audio:
        return "audio"
    if message.photo:
        return "photo"
    if message.document:
        return "document"
    return None


SUPPORTED_FORMATS_DESC: tuple[tuple[str, Iterable[str]], ...] = (
    ("Videos", _VIDEO_EXTS),
    ("Audio", _AUDIO_EXTS),
    ("Archives", _ARCHIVE_EXTS),
    ("Documents", _DOC_EXTS),
    ("Apps", _APP_EXTS),
    ("Images", _IMAGE_EXTS),
    ("Disc Images", _ISO_EXTS),
    ("Subtitles", _SUBTITLE_EXTS),
    ("Telegram native", {"photo", "voice", "video_note", "sticker", "animation"}),
)


# ===========================================================================
# Utils — cleanup
# ===========================================================================


def _safe_remove(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except OSError as exc:
        logger.warning("Could not remove %s: %s", path, exc)


async def remove_path(path) -> None:
    """Asynchronously delete a file or directory tree."""
    if path is None:
        return
    p = Path(path)
    if not p.exists():
        return
    await asyncio.to_thread(_safe_remove, p)


async def cleanup_paths(*paths) -> None:
    """Convenience wrapper to delete many paths concurrently."""
    await asyncio.gather(*(_remove_safe(p) for p in paths), return_exceptions=True)


async def _remove_safe(path) -> None:
    try:
        await remove_path(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("cleanup ignored %s: %s", path, exc)


async def cleanup_directory(directory, max_age_seconds: int) -> int:
    """Delete files inside *directory* older than *max_age_seconds*."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    now = time.time()
    removed = 0
    for entry in directory.iterdir():
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if now - mtime > max_age_seconds:
            await remove_path(entry)
            removed += 1
    return removed


async def periodic_cleanup(downloads_dir, frames_dir, interval: int = 600,
                           max_age: int = 3600) -> None:
    """Background loop that periodically purges stale temp files."""
    logger.info("Periodic cleanup task started (interval=%ss, max_age=%ss)",
                interval, max_age)
    while True:
        try:
            await asyncio.sleep(interval)
            d = await cleanup_directory(downloads_dir, max_age)
            f = await cleanup_directory(frames_dir, max_age)
            if d or f:
                logger.info("Cleanup removed %d downloads, %d frames", d, f)
        except asyncio.CancelledError:
            logger.info("Periodic cleanup task cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Cleanup loop error: %s", exc)


def human_size(num_bytes: int | float) -> str:
    """Human-readable file size."""
    if num_bytes is None:
        return "—"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


# ===========================================================================
# Utils — safe_send (Task ID 6)
# ===========================================================================
# Safe message-sending utilities.
#
# Solves the recurring ``Can't parse entities`` BadRequest errors caused by
# user-generated text (URLs, usernames, captions, filenames, chat titles)
# containing Markdown special characters that Telegram interprets as formatting
# markup.
#
# Public API:
#   * md_escape         — escape Markdown special chars in dynamic text.
#   * html_escape       — escape HTML entities.
#   * safe_send_message / safe_edit_message_text / safe_reply_text /
#     safe_send_document — try Markdown, fall back to plain text on parse error.
#
# Each ``safe_*`` wrapper first tries with ``parse_mode="Markdown"``. If
# Telegram rejects the message with a ``BadRequest`` mentioning entities, it
# automatically retries with ``parse_mode=None`` (plain text) so the user
# always sees something. Transient network / rate-limit errors are also
# swallowed gracefully so a single bad update never crashes an update handler.
# ---------------------------------------------------------------------------

# Markdown v1 special characters (parse_mode="Markdown").
_MD_V1_SPECIALS = ("\\", "`", "*", "_", "[")


def md_escape(text: Any) -> str:
    """Escape Markdown v1 special characters in *dynamic* text.

    Use this on ANY user-generated or variable text before inserting it into a
    Markdown-formatted message: usernames, filenames, captions, chat titles,
    error messages, descriptions, etc.

    Returns ``str(text)`` with ``\\``, ```` ` ````, ``*``, ``_``, ``[`` prefixed
    by a backslash. Static bot text with intentional formatting (``*bold*``)
    should NOT be escaped — only the dynamic insertions.
    """
    if text is None:
        return ""
    s = str(text)
    for ch in _MD_V1_SPECIALS:
        s = s.replace(ch, "\\" + ch)
    return s


def html_escape(text: Any) -> str:
    """Escape HTML special characters (for parse_mode='HTML')."""
    if text is None:
        return ""
    s = str(text)
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def _is_parse_error(exc: Exception) -> bool:
    """True if *exc* is a Markdown/HTML entity-parsing BadRequest."""
    if not isinstance(exc, BadRequest):
        return False
    msg = str(exc).lower()
    return "can't parse entities" in msg or "entity" in msg


async def safe_send_message(
    bot, chat_id, text, *,
    parse_mode: str | None = "Markdown",
    reply_markup=None,
    reply_to_message_id: int | None = None,
    disable_notification: bool = False,
    **kwargs: Any,
):
    """Send a message, falling back to plain text if Markdown parsing fails.

    Also swallows network/rate-limit errors gracefully (logs them) so a
    transient Telegram hiccup never crashes an update handler.
    """
    try:
        return await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            reply_to_message_id=reply_to_message_id,
            disable_notification=disable_notification,
            **kwargs,
        )
    except BadRequest as exc:
        if _is_parse_error(exc):
            logger.warning(
                "Markdown parse failed (%s); retrying as plain text. "
                "Text preview: %.80r", exc, text,
            )
            try:
                return await bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=None,
                    reply_markup=reply_markup,
                    reply_to_message_id=reply_to_message_id,
                    disable_notification=disable_notification,
                    **kwargs,
                )
            except Exception as exc2:  # noqa: BLE001
                logger.error("safe_send_message plain-text fallback failed: %s", exc2)
                return None
        logger.warning("safe_send_message BadRequest: %s", exc)
        return None
    except RetryAfter as exc:
        logger.info("Rate limited on send_message: %ss", exc.retry_after)
        await asyncio.sleep(min(exc.retry_after + 1, 30))
        try:
            return await bot.send_message(
                chat_id=chat_id, text=text, parse_mode=parse_mode,
                reply_markup=reply_markup, reply_to_message_id=reply_to_message_id,
                disable_notification=disable_notification, **kwargs,
            )
        except Exception as exc2:  # noqa: BLE001
            logger.error("safe_send_message retry failed: %s", exc2)
            return None
    except (NetworkError, TimedOut) as exc:
        logger.warning("safe_send_message network error: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("safe_send_message unexpected error: %s", exc)
        return None


async def safe_edit_message_text(
    bot, chat_id, message_id, text, *,
    parse_mode: str | None = "Markdown",
    reply_markup=None,
    **kwargs: Any,
):
    """Edit a message, falling back to plain text if Markdown parsing fails.

    Also tolerates ``MessageNotModified`` and ``Message to edit not found``
    gracefully.
    """
    try:
        return await bot.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=text,
            parse_mode=parse_mode, reply_markup=reply_markup, **kwargs,
        )
    except BadRequest as exc:
        msg = str(exc).lower()
        if _is_parse_error(exc):
            logger.warning(
                "Markdown parse failed on edit (%s); retrying as plain text.", exc
            )
            try:
                return await bot.edit_message_text(
                    chat_id=chat_id, message_id=message_id, text=text,
                    parse_mode=None, reply_markup=reply_markup, **kwargs,
                )
            except Exception as exc2:  # noqa: BLE001
                logger.error("safe_edit plain-text fallback failed: %s", exc2)
                return None
        if "not modified" in msg:
            logger.debug("edit_message_text: not modified (ignored).")
            return None
        if "message to edit not found" in msg or "message is not modified" in msg:
            logger.debug("edit_message_text: message gone (ignored).")
            return None
        logger.warning("safe_edit_message_text BadRequest: %s", exc)
        return None
    except (NetworkError, TimedOut) as exc:
        logger.warning("safe_edit_message_text network error: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("safe_edit_message_text unexpected error: %s", exc)
        return None


async def safe_reply_text(
    message, text, *,
    parse_mode: str | None = "Markdown",
    reply_markup=None,
    quote: bool = False,
    **kwargs: Any,
):
    """Reply to a message, falling back to plain text if Markdown fails."""
    try:
        return await message.reply_text(
            text=text, parse_mode=parse_mode,
            reply_markup=reply_markup, quote=quote, **kwargs,
        )
    except BadRequest as exc:
        if _is_parse_error(exc):
            logger.warning(
                "Markdown parse failed on reply (%s); retrying as plain text.", exc
            )
            try:
                return await message.reply_text(
                    text=text, parse_mode=None,
                    reply_markup=reply_markup, quote=quote, **kwargs,
                )
            except Exception as exc2:  # noqa: BLE001
                logger.error("safe_reply_text plain-text fallback failed: %s", exc2)
                return None
        logger.warning("safe_reply_text BadRequest: %s", exc)
        return None
    except RetryAfter as exc:
        logger.info("Rate limited on reply_text: %ss", exc.retry_after)
        await asyncio.sleep(min(exc.retry_after + 1, 30))
        try:
            return await message.reply_text(
                text=text, parse_mode=parse_mode,
                reply_markup=reply_markup, quote=quote, **kwargs,
            )
        except Exception as exc2:  # noqa: BLE001
            logger.error("safe_reply_text retry failed: %s", exc2)
            return None
    except (NetworkError, TimedOut) as exc:
        logger.warning("safe_reply_text network error: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("safe_reply_text unexpected error: %s", exc)
        return None


async def safe_send_document(
    bot, chat_id, document, *,
    filename: str | None = None,
    caption: str | None = None,
    parse_mode: str | None = "Markdown",
    reply_markup=None,
    **kwargs: Any,
):
    """Send a document, falling back to plain-text caption if Markdown fails."""
    try:
        return await bot.send_document(
            chat_id=chat_id, document=document, filename=filename,
            caption=caption, parse_mode=parse_mode,
            reply_markup=reply_markup, **kwargs,
        )
    except BadRequest as exc:
        if _is_parse_error(exc):
            logger.warning(
                "Markdown parse failed on document caption (%s); "
                "retrying as plain text.", exc
            )
            try:
                return await bot.send_document(
                    chat_id=chat_id, document=document, filename=filename,
                    caption=caption, parse_mode=None,
                    reply_markup=reply_markup, **kwargs,
                )
            except Exception as exc2:  # noqa: BLE001
                logger.error("safe_send_document fallback failed: %s", exc2)
                return None
        logger.warning("safe_send_document BadRequest: %s", exc)
        return None
    except (NetworkError, TimedOut) as exc:
        logger.warning("safe_send_document network error: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("safe_send_document unexpected error: %s", exc)
        return None


# ===========================================================================
# Utils — bot_registry (Task ID 10)
# ===========================================================================
# Stores a reference to the running ``telegram.Bot`` instance so that
# background services (like the MTProto self-destruct capture handler) can
# send messages without needing access to the PTB ``Application`` object
# (which has no public ``get_application()`` singleton accessor).
#
# Usage:
#   * In ``_post_init``:  ``register_bot(app.bot)``
#   * In background code: ``bot = get_bot()`` (may be None until registered)


_bot_instance: Optional[Any] = None


def register_bot(bot) -> None:
    """Store a reference to the running Bot instance."""
    global _bot_instance
    _bot_instance = bot


def get_bot():
    """Return the running Bot instance, or None if not registered yet."""
    return _bot_instance


def is_available() -> bool:
    """True if a Bot instance has been registered."""
    return _bot_instance is not None


# ===========================================================================
# Database — schema
# ===========================================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    tg_id            INTEGER PRIMARY KEY,
    username         TEXT,
    first_name       TEXT,
    is_admin         INTEGER DEFAULT 0,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    last_active      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id          INTEGER PRIMARY KEY REFERENCES users(tg_id) ON DELETE CASCADE,
    preferred_quality TEXT NOT NULL DEFAULT '1080p',
    ai_model         TEXT NOT NULL DEFAULT 'gemini-2.0-flash',
    language         TEXT NOT NULL DEFAULT 'English',
    auto_delete      INTEGER NOT NULL DEFAULT 1,
    notifications    INTEGER NOT NULL DEFAULT 1,
    ai_mode          TEXT NOT NULL DEFAULT 'movie',
    gemini_api_key   TEXT,
    toolbox_audio_bitrate TEXT NOT NULL DEFAULT '192k',
    toolbox_video_crf INTEGER NOT NULL DEFAULT 28,
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tasks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
    kind             TEXT NOT NULL,
    status           TEXT NOT NULL,
    progress         INTEGER NOT NULL DEFAULT 0,
    error            TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS download_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
    task_id          INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    file_name        TEXT NOT NULL,
    file_size        INTEGER NOT NULL DEFAULT 0,
    mime_type        TEXT,
    media_type       TEXT,
    source           TEXT NOT NULL,
    status           TEXT NOT NULL,
    file_unique_id   TEXT,
    message_link     TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_dl_user ON download_history(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ai_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
    task_id          INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    file_name        TEXT NOT NULL,
    media_type       TEXT,
    category         TEXT,
    title            TEXT,
    season           TEXT,
    episode          TEXT,
    year             TEXT,
    language         TEXT,
    quality          TEXT,
    confidence       REAL,
    raw_json         TEXT,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ai_user ON ai_history(user_id, created_at DESC);

-- Saved / bookmarked media library
CREATE TABLE IF NOT EXISTS library (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
    file_name        TEXT NOT NULL,
    media_type       TEXT,
    file_size        INTEGER NOT NULL DEFAULT 0,
    file_id          TEXT,
    note             TEXT,
    tags             TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_lib_user ON library(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lib_name ON library(file_name);

-- Inspected chats / users (the "finder" history)
CREATE TABLE IF NOT EXISTS inspected_chats (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
    chat_id          INTEGER,
    username         TEXT,
    title            TEXT,
    chat_type        TEXT,
    members          INTEGER,
    description      TEXT,
    first_name       TEXT,
    last_name        TEXT,
    bio              TEXT,
    is_bot           INTEGER,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_insp_user ON inspected_chats(user_id, created_at DESC);

-- Scheduled download queue
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
    kind             TEXT NOT NULL,
    payload          TEXT NOT NULL,
    run_at           TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sched_run ON scheduled_tasks(run_at, status);

-- Admin broadcasts
CREATE TABLE IF NOT EXISTS broadcasts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_id         INTEGER NOT NULL,
    text             TEXT NOT NULL,
    sent             INTEGER NOT NULL DEFAULT 0,
    failed           INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- VC Tour: discovered groups
CREATE TABLE IF NOT EXISTS vc_groups (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id         INTEGER NOT NULL UNIQUE,
    access_hash      INTEGER,
    title            TEXT,
    username         TEXT,
    public_link      TEXT,
    source           TEXT,                 -- dialog|explicit|config
    access_status    TEXT NOT NULL DEFAULT 'unknown', -- accessible|inaccessible|banned|restricted
    active_vc        INTEGER NOT NULL DEFAULT 0,
    discovered_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_checked_at  TEXT,
    last_joined_at   TEXT,
    last_error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_vc_groups_active ON vc_groups(active_vc);

-- VC Tour: visit history
CREATE TABLE IF NOT EXISTS vc_visits (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id         INTEGER NOT NULL,
    group_title      TEXT,
    username         TEXT,
    group_link       TEXT,
    joined_at        TEXT,
    left_at          TEXT,
    planned_duration_seconds INTEGER,
    actual_duration_seconds  INTEGER,
    mode             TEXT,                 -- auto|manual
    status           TEXT,                 -- joined|completed|failed|disconnected
    leave_reason     TEXT,
    error            TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_vc_visits_group ON vc_visits(group_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vc_visits_recent ON vc_visits(created_at DESC);

-- VC Tour: persisted tour state (single row, id=1)
CREATE TABLE IF NOT EXISTS vc_tour_state (
    id               INTEGER PRIMARY KEY DEFAULT 1,
    running          INTEGER NOT NULL DEFAULT 0,
    paused           INTEGER NOT NULL DEFAULT 0,
    current_group_id INTEGER,
    current_queue_index INTEGER NOT NULL DEFAULT 0,
    queue_json       TEXT,
    started_at       TEXT,
    updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
    stay_seconds     INTEGER NOT NULL DEFAULT 300,
    cooldown_seconds INTEGER NOT NULL DEFAULT 30
);
INSERT OR IGNORE INTO vc_tour_state (id, running, paused, stay_seconds, cooldown_seconds)
VALUES (1, 0, 0, 300, 30);
"""

MIGRATION_SQL = [
    "ALTER TABLE user_settings ADD COLUMN ai_mode TEXT NOT NULL DEFAULT 'movie'",
    "ALTER TABLE user_settings ADD COLUMN gemini_api_key TEXT",
    "ALTER TABLE user_settings ADD COLUMN toolbox_audio_bitrate TEXT NOT NULL DEFAULT '192k'",
    "ALTER TABLE user_settings ADD COLUMN toolbox_video_crf INTEGER NOT NULL DEFAULT 28",
]


# ===========================================================================
# Database — connection
# ===========================================================================

_db: aiosqlite.Connection | None = None


async def _column_exists(db, table: str, column: str) -> bool:
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        rows = await cur.fetchall()
    return any(row[1] == column for row in rows)


async def _run_migrations(db) -> None:
    """Add new columns to existing tables (idempotent)."""
    for stmt in MIGRATION_SQL:
        m = re.search(r"ADD COLUMN\s+(\w+)", stmt, re.IGNORECASE)
        if not m:
            continue
        col = m.group(1)
        if not await _column_exists(db, "user_settings", col):
            try:
                await db.execute(stmt)
                logger.info("Migration: added column %s to user_settings", col)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Migration skipped (%s): %s", col, exc)
    await db.commit()


async def init_db() -> aiosqlite.Connection:
    """Initialise the database connection and create tables."""
    global _db
    if _db is not None:
        return _db
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    _db = await aiosqlite.connect(str(config.db_path))
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL;")
    await _db.execute("PRAGMA synchronous=NORMAL;")
    await _db.execute("PRAGMA foreign_keys=ON;")
    await _db.executescript(SCHEMA_SQL)
    await _run_migrations(_db)
    await _db.commit()
    logger.info("Database ready at %s", config.db_path)
    return _db


async def get_db() -> aiosqlite.Connection:
    if _db is None:
        return await init_db()
    return _db


async def close_db() -> None:
    global _db
    if _db is not None:
        await _db.close()
        _db = None
        logger.info("Database connection closed")


# ===========================================================================
# Database — repositories
# ===========================================================================


async def upsert_user(tg_id: int, username: str | None, first_name: str | None,
                      is_admin: bool = False) -> None:
    db = await get_db()
    await db.execute(
        """
        INSERT INTO users (tg_id, username, first_name, is_admin, last_active)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(tg_id) DO UPDATE SET
            username = excluded.username,
            first_name = excluded.first_name,
            is_admin = excluded.is_admin,
            last_active = datetime('now')
        """,
        (tg_id, username, first_name, 1 if is_admin else 0),
    )
    await db.commit()


async def ensure_settings(tg_id: int) -> dict[str, Any]:
    """Create default settings row if missing and return current settings."""
    db = await get_db()
    await db.execute(
        """
        INSERT OR IGNORE INTO user_settings
            (user_id, preferred_quality, ai_model, language, auto_delete, notifications)
        VALUES (?, ?, ?, ?, 1, 1)
        """,
        (tg_id, config.default_quality, config.default_ai_model, config.default_language),
    )
    await db.commit()
    async with db.execute(
        "SELECT * FROM user_settings WHERE user_id = ?", (tg_id,)
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row else {}


async def update_setting(tg_id: int, column: str, value: Any) -> dict[str, Any]:
    """Update a single settings column safely and return the new row."""
    allowed = {
        "preferred_quality", "ai_model", "language",
        "auto_delete", "notifications",
        "ai_mode", "gemini_api_key",
        "toolbox_audio_bitrate", "toolbox_video_crf",
    }
    if column not in allowed:
        raise ValueError(f"Cannot update unknown setting: {column}")
    await ensure_settings(tg_id)
    db = await get_db()
    await db.execute(
        f"UPDATE user_settings SET {column} = ?, updated_at = datetime('now') "
        f"WHERE user_id = ?",
        (value, tg_id),
    )
    await db.commit()
    return await ensure_settings(tg_id)


async def create_task(user_id: int, kind: str, status: str = "pending") -> int:
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO tasks (user_id, kind, status) VALUES (?, ?, ?)",
        (user_id, kind, status),
    )
    await db.commit()
    return cur.lastrowid  # type: ignore[return-value]


async def update_task(task_id: int, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    values: list[Any] = list(fields.values())
    values.append(task_id)
    db = await get_db()
    await db.execute(
        f"UPDATE tasks SET {cols}, updated_at = datetime('now') WHERE id = ?",
        values,
    )
    await db.commit()


async def add_download_history(
    user_id: int, *, file_name: str, file_size: int, mime_type: str | None,
    media_type: str | None, source: str, status: str,
    task_id: int | None = None, file_unique_id: str | None = None,
    message_link: str | None = None,
) -> int:
    db = await get_db()
    cur = await db.execute(
        """
        INSERT INTO download_history
            (user_id, task_id, file_name, file_size, mime_type, media_type,
             source, status, file_unique_id, message_link)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, task_id, file_name, file_size, mime_type, media_type,
         source, status, file_unique_id, message_link),
    )
    await db.commit()
    return cur.lastrowid  # type: ignore[return-value]


async def recent_downloads(user_id: int, limit: int = 10) -> list[dict[str, Any]]:
    db = await get_db()
    async with db.execute(
        """
        SELECT file_name, file_size, media_type, source, status, created_at
        FROM download_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def add_ai_history(
    user_id: int, *, file_name: str, media_type: str | None,
    result: dict[str, Any] | None, status: str,
    task_id: int | None = None,
) -> int:
    db = await get_db()
    cur = await db.execute(
        """
        INSERT INTO ai_history
            (user_id, task_id, file_name, media_type, category, title,
             season, episode, year, language, quality, confidence, raw_json, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id, task_id, file_name, media_type,
            (result or {}).get("category"),
            (result or {}).get("title"),
            (result or {}).get("season"),
            (result or {}).get("episode"),
            (result or {}).get("year"),
            (result or {}).get("language"),
            (result or {}).get("quality"),
            (result or {}).get("confidence"),
            json.dumps(result) if result else None,
            status,
        ),
    )
    await db.commit()
    return cur.lastrowid  # type: ignore[return-value]


async def recent_ai_analyses(user_id: int, limit: int = 10) -> list[dict[str, Any]]:
    db = await get_db()
    async with db.execute(
        """
        SELECT file_name, media_type, category, title, season, episode, year,
               language, confidence, status, created_at
        FROM ai_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def clear_history(user_id: int, kind: str | None = None) -> int:
    """Delete history rows for a user. kind: 'download'|'ai'|None (both)."""
    db = await get_db()
    count = 0
    if kind in (None, "download"):
        cur = await db.execute(
            "DELETE FROM download_history WHERE user_id = ?", (user_id,)
        )
        count += cur.rowcount or 0
    if kind in (None, "ai"):
        cur = await db.execute(
            "DELETE FROM ai_history WHERE user_id = ?", (user_id,)
        )
        count += cur.rowcount or 0
    await db.commit()
    logger.info("Cleared %d history rows for user %s (kind=%s)", count, user_id, kind)
    return count


async def stats() -> dict[str, int]:
    db = await get_db()
    out: dict[str, int] = {}
    async with db.execute("SELECT COUNT(*) FROM users") as cur:
        out["users"] = (await cur.fetchone())[0]
    async with db.execute("SELECT COUNT(*) FROM download_history WHERE status='done'") as cur:
        out["downloads"] = (await cur.fetchone())[0]
    async with db.execute("SELECT COUNT(*) FROM ai_history WHERE status='done'") as cur:
        out["analyses"] = (await cur.fetchone())[0]
    return out


async def fetch_all(sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    db = await get_db()
    async with db.execute(sql, tuple(params)) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def add_library_entry(
    user_id: int, *, file_name: str, media_type: str | None,
    file_size: int, file_id: str | None = None, note: str | None = None,
    tags: str | None = None,
) -> int:
    db = await get_db()
    cur = await db.execute(
        """
        INSERT INTO library (user_id, file_name, media_type, file_size, file_id, note, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, file_name, media_type, file_size, file_id, note, tags),
    )
    await db.commit()
    return cur.lastrowid  # type: ignore[return-value]


async def library_entries(user_id: int, limit: int = 20, offset: int = 0,
                          media_type: str | None = None) -> list[dict[str, Any]]:
    db = await get_db()
    if media_type:
        sql = ("SELECT * FROM library WHERE user_id = ? AND media_type = ? "
               "ORDER BY created_at DESC LIMIT ? OFFSET ?")
        params: tuple[Any, ...] = (user_id, media_type, limit, offset)
    else:
        sql = ("SELECT * FROM library WHERE user_id = ? "
               "ORDER BY created_at DESC LIMIT ? OFFSET ?")
        params = (user_id, limit, offset)
    async with db.execute(sql, params) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def library_search(user_id: int, query: str, limit: int = 20) -> list[dict[str, Any]]:
    db = await get_db()
    like = f"%{query.lower()}%"
    async with db.execute(
        """
        SELECT * FROM library
        WHERE user_id = ? AND (
            LOWER(file_name) LIKE ? OR LOWER(IFNULL(note,'')) LIKE ?
            OR LOWER(IFNULL(tags,'')) LIKE ?
        )
        ORDER BY created_at DESC LIMIT ?
        """,
        (user_id, like, like, like, limit),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def library_count(user_id: int) -> int:
    db = await get_db()
    async with db.execute(
        "SELECT COUNT(*) FROM library WHERE user_id = ?", (user_id,)
    ) as cur:
        return (await cur.fetchone())[0]


async def library_remove(user_id: int, entry_id: int) -> bool:
    db = await get_db()
    cur = await db.execute(
        "DELETE FROM library WHERE id = ? AND user_id = ?", (entry_id, user_id)
    )
    await db.commit()
    return (cur.rowcount or 0) > 0


async def library_clear(user_id: int) -> int:
    db = await get_db()
    cur = await db.execute("DELETE FROM library WHERE user_id = ?", (user_id,))
    await db.commit()
    return cur.rowcount or 0


async def add_inspected_chat(
    user_id: int, *, chat_id: int | None, username: str | None,
    title: str | None, chat_type: str | None, members: int | None,
    description: str | None, first_name: str | None, last_name: str | None,
    bio: str | None, is_bot: bool | None,
) -> int:
    db = await get_db()
    cur = await db.execute(
        """
        INSERT INTO inspected_chats
            (user_id, chat_id, username, title, chat_type, members,
             description, first_name, last_name, bio, is_bot)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, chat_id, username, title, chat_type, members, description,
         first_name, last_name, bio, 1 if is_bot else 0 if is_bot is False else None),
    )
    await db.commit()
    return cur.lastrowid  # type: ignore[return-value]


async def recent_inspected(user_id: int, limit: int = 10) -> list[dict[str, Any]]:
    db = await get_db()
    async with db.execute(
        """
        SELECT username, title, chat_type, members, first_name, is_bot, created_at
        FROM inspected_chats WHERE user_id = ?
        ORDER BY created_at DESC LIMIT ?
        """,
        (user_id, limit),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def user_daily_download_usage(user_id: int) -> tuple[int, int]:
    """Return (count_today, bytes_today) of completed downloads."""
    db = await get_db()
    async with db.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(file_size), 0)
        FROM download_history
        WHERE user_id = ? AND status = 'done'
          AND date(created_at) = date('now')
        """,
        (user_id,),
    ) as cur:
        row = await cur.fetchone()
    return int(row[0] or 0), int(row[1] or 0)


async def user_stats(user_id: int) -> dict[str, Any]:
    db = await get_db()
    out: dict[str, Any] = {}
    async with db.execute(
        "SELECT COUNT(*), COALESCE(SUM(file_size),0) FROM download_history "
        "WHERE user_id = ? AND status='done'", (user_id,),
    ) as cur:
        c, b = await cur.fetchone()
        out["downloads"] = int(c or 0)
        out["download_bytes"] = int(b or 0)
    async with db.execute(
        "SELECT COUNT(*) FROM ai_history WHERE user_id = ? AND status='done'",
        (user_id,),
    ) as cur:
        out["analyses"] = (await cur.fetchone())[0]
    async with db.execute(
        "SELECT COUNT(*) FROM library WHERE user_id = ?", (user_id,),
    ) as cur:
        out["library"] = (await cur.fetchone())[0]
    async with db.execute(
        "SELECT COUNT(*) FROM inspected_chats WHERE user_id = ?", (user_id,),
    ) as cur:
        out["inspected"] = (await cur.fetchone())[0]
    async with db.execute(
        "SELECT media_type, COUNT(*) FROM download_history "
        "WHERE user_id = ? AND status='done' GROUP BY media_type "
        "ORDER BY COUNT(*) DESC LIMIT 5", (user_id,),
    ) as cur:
        out["by_type"] = [(r[0], r[1]) for r in await cur.fetchall()]
    return out


async def global_stats() -> dict[str, Any]:
    db = await get_db()
    out: dict[str, Any] = {}
    async with db.execute("SELECT COUNT(*) FROM users") as cur:
        out["users"] = (await cur.fetchone())[0]
    async with db.execute(
        "SELECT COUNT(*), COALESCE(SUM(file_size),0) FROM download_history "
        "WHERE status='done'"
    ) as cur:
        c, b = await cur.fetchone()
        out["downloads"] = int(c or 0)
        out["download_bytes"] = int(b or 0)
    async with db.execute(
        "SELECT COUNT(*) FROM ai_history WHERE status='done'"
    ) as cur:
        out["analyses"] = (await cur.fetchone())[0]
    async with db.execute("SELECT COUNT(*) FROM library") as cur:
        out["library"] = (await cur.fetchone())[0]
    async with db.execute("SELECT COUNT(*) FROM inspected_chats") as cur:
        out["inspected"] = (await cur.fetchone())[0]
    return out


async def add_scheduled_task(user_id: int, kind: str, payload: str,
                             run_at: str) -> int:
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO scheduled_tasks (user_id, kind, payload, run_at) "
        "VALUES (?, ?, ?, ?)",
        (user_id, kind, payload, run_at),
    )
    await db.commit()
    return cur.lastrowid  # type: ignore[return-value]


async def pending_scheduled_tasks(now_iso: str, limit: int = 20) -> list[dict[str, Any]]:
    db = await get_db()
    async with db.execute(
        "SELECT * FROM scheduled_tasks WHERE status='pending' AND run_at <= ? "
        "ORDER BY run_at ASC LIMIT ?",
        (now_iso, limit),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def list_scheduled_tasks(user_id: int, limit: int = 10) -> list[dict[str, Any]]:
    db = await get_db()
    async with db.execute(
        "SELECT id, kind, payload, run_at, status, created_at "
        "FROM scheduled_tasks WHERE user_id = ? ORDER BY run_at DESC LIMIT ?",
        (user_id, limit),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def update_scheduled_task(task_id: int, status: str) -> None:
    db = await get_db()
    await db.execute(
        "UPDATE scheduled_tasks SET status = ? WHERE id = ?", (status, task_id)
    )
    await db.commit()


async def cancel_scheduled_task(user_id: int, task_id: int) -> bool:
    db = await get_db()
    cur = await db.execute(
        "DELETE FROM scheduled_tasks WHERE id = ? AND user_id = ? AND status='pending'",
        (task_id, user_id),
    )
    await db.commit()
    return (cur.rowcount or 0) > 0


async def create_broadcast(admin_id: int, text: str) -> int:
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO broadcasts (admin_id, text) VALUES (?, ?)",
        (admin_id, text),
    )
    await db.commit()
    return cur.lastrowid  # type: ignore[return-value]


async def update_broadcast_counts(broadcast_id: int, sent: int, failed: int) -> None:
    db = await get_db()
    await db.execute(
        "UPDATE broadcasts SET sent = ?, failed = ? WHERE id = ?",
        (sent, failed, broadcast_id),
    )
    await db.commit()


async def recent_broadcasts(limit: int = 5) -> list[dict[str, Any]]:
    db = await get_db()
    async with db.execute(
        "SELECT * FROM broadcasts ORDER BY created_at DESC LIMIT ?", (limit,)
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def all_users(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    db = await get_db()
    async with db.execute(
        "SELECT tg_id, username, first_name, is_admin, created_at, last_active "
        "FROM users ORDER BY last_active DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def user_count() -> int:
    db = await get_db()
    async with db.execute("SELECT COUNT(*) FROM users") as cur:
        return (await cur.fetchone())[0]


async def export_user_data(user_id: int) -> dict[str, Any]:
    """Export a user's settings, history, library, and inspections as a dict."""
    db = await get_db()
    out: dict[str, Any] = {"user_id": user_id}
    async with db.execute(
        "SELECT * FROM user_settings WHERE user_id = ?", (user_id,)
    ) as cur:
        s = await cur.fetchone()
    out["settings"] = dict(s) if s else {}
    if out["settings"].get("gemini_api_key"):
        out["settings"]["gemini_api_key"] = "***redacted***"
    async with db.execute(
        "SELECT file_name, file_size, media_type, source, status, created_at "
        "FROM download_history WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ) as cur:
        out["download_history"] = [dict(r) for r in await cur.fetchall()]
    async with db.execute(
        "SELECT file_name, category, title, season, episode, year, language, "
        "quality, confidence, status, created_at FROM ai_history "
        "WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ) as cur:
        out["ai_history"] = [dict(r) for r in await cur.fetchall()]
    out["library"] = await library_entries(user_id, limit=1000)
    async with db.execute(
        "SELECT username, title, chat_type, members, created_at "
        "FROM inspected_chats WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ) as cur:
        out["inspected_chats"] = [dict(r) for r in await cur.fetchall()]
    return out


async def restore_user_settings(user_id: int, settings: dict[str, Any]) -> None:
    """Import settings from a backup dict (only safe, known columns)."""
    if not settings:
        return
    await ensure_settings(user_id)
    db = await get_db()
    safe = {
        "preferred_quality", "ai_model", "language", "auto_delete",
        "notifications", "ai_mode", "toolbox_audio_bitrate",
        "toolbox_video_crf",
    }
    sets: list[str] = []
    vals: list[Any] = []
    for k, v in settings.items():
        if k in safe and v is not None:
            sets.append(f"{k} = ?")
            vals.append(v)
    if sets:
        vals.append(user_id)
        await db.execute(
            f"UPDATE user_settings SET {', '.join(sets)}, "
            f"updated_at = datetime('now') WHERE user_id = ?",
            vals,
        )
        await db.commit()


# ---------------------------------------------------------------------------
# VC Tour — groups, visits, tour state
# ---------------------------------------------------------------------------


async def vc_upsert_group(*, group_id: int, access_hash: int | None = None,
                          title: str | None = None, username: str | None = None,
                          public_link: str | None = None, source: str | None = None,
                          access_status: str = "unknown",
                          active_vc: bool = False,
                          last_error: str | None = None) -> None:
    db = await get_db()
    await db.execute(
        """
        INSERT INTO vc_groups (group_id, access_hash, title, username,
                                public_link, source, access_status, active_vc,
                                discovered_at, last_checked_at, last_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), ?)
        ON CONFLICT(group_id) DO UPDATE SET
            access_hash = excluded.access_hash,
            title = excluded.title,
            username = excluded.username,
            public_link = excluded.public_link,
            source = excluded.source,
            access_status = excluded.access_status,
            active_vc = excluded.active_vc,
            last_checked_at = datetime('now'),
            last_error = excluded.last_error
        """,
        (group_id, access_hash, title, username, public_link, source,
         access_status, 1 if active_vc else 0, last_error),
    )
    await db.commit()


async def vc_groups_active(limit: int = 500) -> list[dict[str, Any]]:
    db = await get_db()
    async with db.execute(
        "SELECT * FROM vc_groups WHERE active_vc=1 AND access_status='accessible' "
        "ORDER BY last_checked_at DESC LIMIT ?",
        (limit,),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def vc_groups_all(limit: int = 500) -> list[dict[str, Any]]:
    db = await get_db()
    async with db.execute(
        "SELECT * FROM vc_groups ORDER BY discovered_at DESC LIMIT ?", (limit,)
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def vc_add_visit(*, group_id: int, group_title: str | None = None,
                       username: str | None = None, group_link: str | None = None,
                       joined_at: str | None = None, left_at: str | None = None,
                       planned_duration_seconds: int | None = None,
                       actual_duration_seconds: int | None = None,
                       mode: str = "auto", status: str = "joined",
                       leave_reason: str | None = None,
                       error: str | None = None) -> int:
    db = await get_db()
    cur = await db.execute(
        """
        INSERT INTO vc_visits (group_id, group_title, username, group_link,
                               joined_at, left_at, planned_duration_seconds,
                               actual_duration_seconds, mode, status,
                               leave_reason, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (group_id, group_title, username, group_link, joined_at, left_at,
         planned_duration_seconds, actual_duration_seconds, mode, status,
         leave_reason, error),
    )
    await db.commit()
    return cur.lastrowid  # type: ignore[return-value]


async def vc_update_visit(visit_id: int, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals: list[Any] = list(fields.values())
    vals.append(visit_id)
    db = await get_db()
    await db.execute(f"UPDATE vc_visits SET {cols} WHERE id = ?", vals)
    await db.commit()


async def vc_recent_visits(limit: int = 20, offset: int = 0,
                           mode: str | None = None,
                           status: str | None = None) -> list[dict[str, Any]]:
    db = await get_db()
    clauses = []
    params: list[Any] = []
    if mode:
        clauses.append("mode = ?")
        params.append(mode)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.extend([limit, offset])
    async with db.execute(
        f"SELECT * FROM vc_visits {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params,
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def vc_visit_count(mode: str | None = None,
                         status: str | None = None) -> int:
    db = await get_db()
    clauses = []
    params: list[Any] = []
    if mode:
        clauses.append("mode = ?")
        params.append(mode)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    async with db.execute(f"SELECT COUNT(*) FROM vc_visits {where}", params) as cur:
        return (await cur.fetchone())[0]


async def vc_get_tour_state() -> dict[str, Any]:
    db = await get_db()
    async with db.execute("SELECT * FROM vc_tour_state WHERE id = 1") as cur:
        row = await cur.fetchone()
    return dict(row) if row else {}


async def vc_save_tour_state(**fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = "datetime('now')"
    cols_parts = []
    vals: list[Any] = []
    for k, v in fields.items():
        if v == "datetime('now')":
            cols_parts.append(f"{k} = datetime('now')")
        else:
            cols_parts.append(f"{k} = ?")
            vals.append(v)
    vals.append(1)
    db = await get_db()
    await db.execute(
        f"UPDATE vc_tour_state SET {', '.join(cols_parts)} WHERE id = ?", vals
    )
    await db.commit()


async def vc_visited_group_ids_since(since_iso: str) -> set[int]:
    """Return group IDs visited since *since_iso* (for revisit dedup)."""
    db = await get_db()
    async with db.execute(
        "SELECT DISTINCT group_id FROM vc_visits WHERE created_at >= ?",
        (since_iso,),
    ) as cur:
        rows = await cur.fetchall()
    return {r[0] for r in rows}


# ===========================================================================
# Services — i18n
# ===========================================================================

_EN: dict[str, str] = {
    "main_menu": "🏠 Main Menu\n\nChoose what you'd like to do:",
    "download": "📥 Download",
    "ai_analyze": "🎬 AI Analyze",
    "inspect": "🔍 Inspect Chat",
    "toolbox": "🧰 Media Toolbox",
    "library": "⭐ Library",
    "stats": "📊 Stats",
    "qr": "🔳 QR Code",
    "history": "📜 History",
    "settings": "⚙️ Settings",
    "help": "ℹ️ Help",
    "batch": "📦 Batch",
    "scheduled": "⏰ Scheduled",
    "admin": "🛡️ Admin",
    "backup": "💾 Backup",
    "back": "🔙 Back",
    "cancel": "❌ Cancel",
    "done": "✅ Done",
    "welcome": (
        "👋 Welcome to MediaGrab AI Bot\n\n"
        "Your all-in-one assistant for media downloading, AI analysis and more.\n\n"
        "Tap a button below to begin — no commands needed."
    ),
}

_HI: dict[str, str] = {
    "main_menu": "🏠 मुख्य मेनू\n\nआप क्या करना चाहते हैं चुनें:",
    "download": "📥 डाउनलोड",
    "ai_analyze": "🎬 एआई विश्लेषण",
    "back": "🔙 वापस",
    "cancel": "❌ रद्द करें",
    "done": "✅ पूर्ण",
    "welcome": "👋 MediaGrab AI Bot में आपका स्वागत है\n\nशुरू करने के लिए नीचे एक बटन टैप करें।",
}

_ES: dict[str, str] = {
    "main_menu": "🏠 Menú principal\n\nElige qué quieres hacer:",
    "download": "📥 Descargar",
    "back": "🔙 Atrás",
    "cancel": "❌ Cancelar",
    "done": "✅ Hecho",
}

_FR: dict[str, str] = {
    "main_menu": "🏠 Menu principal\n\nChoisissez une action :",
    "download": "📥 Télécharger",
    "back": "🔙 Retour",
    "cancel": "❌ Annuler",
    "done": "✅ Terminé",
}

_AR: dict[str, str] = {
    "main_menu": "🏠 القائمة الرئيسية\n\nاختر ما تريد فعله:",
    "download": "📥 تحميل",
    "back": "🔙 رجوع",
    "cancel": "❌ إلغاء",
    "done": "✅ تم",
}

_ZH: dict[str, str] = {
    "main_menu": "🏠 主菜单\n\n请选择您要执行的操作：",
    "download": "📥 下载",
    "back": "🔙 返回",
    "cancel": "❌ 取消",
    "done": "✅ 完成",
}

_PT: dict[str, str] = {
    "main_menu": "🏠 Menu principal\n\nEscolha o que deseja fazer:",
    "download": "📥 Baixar",
    "back": "🔙 Voltar",
    "cancel": "❌ Cancelar",
    "done": "✅ Concluído",
}

_LANGS: dict[str, Mapping[str, str]] = {
    "English": _EN,
    "हिन्दी": _HI,
    "Español": _ES,
    "français": _FR,
    "العربية": _AR,
    "中文": _ZH,
    "português": _PT,
}


def supported_languages() -> tuple[str, ...]:
    return tuple(_LANGS.keys())


def t(language: str | None, key: str) -> str:
    """Translate ``key`` for the given language; fall back to English."""
    table = _LANGS.get(language or "English", _EN)
    return table.get(key, _EN.get(key, key))


def has_language(lang: str) -> bool:
    return lang in _LANGS


# ===========================================================================
# Services — ai_modes
# ===========================================================================

ACCEPTS = {
    "movie": {"video"},
    "transcribe": {"audio", "video"},
    "ocr": {"image", "video"},
    "describe": {"image", "video"},
    "translate": {"image", "video"},
}

MODE_LABELS: dict[str, str] = {
    "movie": "🎬 Movie / Series ID",
    "transcribe": "🎵 Audio Transcription",
    "ocr": "🔤 Text Extraction (OCR)",
    "describe": "📝 Scene Description",
    "translate": "🌐 Image Translation",
}

MODE_ORDER = ("movie", "transcribe", "ocr", "describe", "translate")


_MOVIE_PROMPT = """\
You are a media identification expert. You will receive several frames
extracted from a video file. Identify the source as precisely as possible.

Respond ONLY with a single JSON object (no markdown, no explanation) with
exactly these keys:

{
  "category": "Movie" | "TV Series" | "Anime" | "Unknown",
  "title":    the title of the movie, series or anime,
  "season":   season number as a string (e.g. "2") or "" if not applicable,
  "episode":  episode number or title as a string (e.g. "5" or "The One...") or "",
  "year":     release year as a string (e.g. "2019") or "" if unknown,
  "language": the primary spoken/written language (e.g. "English", "Japanese") or "",
  "quality":  the video quality you can infer from the frames (e.g. "720p", "1080p", "4K") or "",
  "confidence": your confidence score from 0.0 to 1.0
}

Rules:
- If you cannot identify the media, set category to "Unknown", leave other
  fields empty strings, and confidence low.
- Use the visible text (titles, subtitles, channel logos) as primary evidence.
- Never wrap the JSON in code fences or add commentary."""

_TRANSCRIBE_PROMPT = """\
You are an expert transcription AI. You will receive an audio file.
Transcribe all speech you can hear, accurately and in the original language.

Respond ONLY with a single JSON object:
{
  "language": the detected spoken language (e.g. "English"),
  "text":     the full transcription,
  "confidence": your confidence from 0.0 to 1.0
}
If there is no speech, set text to "" and confidence to 0.0. Do not add
commentary or code fences."""

_OCR_PROMPT = """\
You are an OCR expert. You will receive one or more images.
Extract ALL visible text exactly as it appears, preserving line breaks.

Respond ONLY with a single JSON object:
{
  "language": the detected script/language of the text (e.g. "English"),
  "text":     all extracted text, newline-separated,
  "confidence": your confidence from 0.0 to 1.0
}
If there is no text, set text to "" and confidence to 0.0. No commentary."""

_DESCRIBE_PROMPT = """\
You are an image description expert. You will receive one or more images
(or frames from a video).
Describe what you see in clear, concise detail: setting, subjects, actions,
text, mood, and any notable visual elements.

Respond ONLY with a single JSON object:
{
  "summary":   a one-sentence summary,
  "details":   a longer paragraph description,
  "tags":      a comma-separated list of relevant tags,
  "confidence": your confidence from 0.0 to 1.0
}
No commentary or code fences."""

_TRANSLATE_PROMPT = """\
You are a translation expert. You will receive one or more images containing
text. First read the text via OCR, then translate it into the target language.

Respond ONLY with a single JSON object:
{
  "source_language": the detected source language,
  "source_text":     the original text exactly as shown,
  "translated_text": the translation into the target language,
  "confidence":      your confidence from 0.0 to 1.0
}
If there is no text, set both text fields to "" and confidence to 0.0.
No commentary or code fences."""


PROMPTS: dict[str, str] = {
    "movie": _MOVIE_PROMPT,
    "transcribe": _TRANSCRIBE_PROMPT,
    "ocr": _OCR_PROMPT,
    "describe": _DESCRIBE_PROMPT,
    "translate": _TRANSLATE_PROMPT,
}


def build_prompt(mode: str, *, target_language: str | None = None) -> str:
    """Return the prompt for the given mode, with mode-specific substitutions."""
    base = PROMPTS.get(mode, _DESCRIBE_PROMPT)
    if mode == "translate" and target_language:
        return base.replace(
            "translate it into the target language.",
            f"translate it into {target_language}.",
        )
    return base


def _norm_movie(data: dict) -> dict[str, Any]:
    return {
        "category": data.get("category", "Unknown"),
        "title": data.get("title", ""),
        "season": data.get("season", ""),
        "episode": data.get("episode", ""),
        "year": data.get("year", ""),
        "language": data.get("language", ""),
        "quality": data.get("quality", ""),
        "confidence": _conf(data.get("confidence")),
    }


def _norm_transcribe(data: dict) -> dict[str, Any]:
    return {
        "language": data.get("language", ""),
        "text": data.get("text", ""),
        "confidence": _conf(data.get("confidence")),
    }


def _norm_ocr(data: dict) -> dict[str, Any]:
    return {
        "language": data.get("language", ""),
        "text": data.get("text", ""),
        "confidence": _conf(data.get("confidence")),
    }


def _norm_describe(data: dict) -> dict[str, Any]:
    return {
        "summary": data.get("summary", ""),
        "details": data.get("details", ""),
        "tags": data.get("tags", ""),
        "confidence": _conf(data.get("confidence")),
    }


def _norm_translate(data: dict) -> dict[str, Any]:
    return {
        "source_language": data.get("source_language", ""),
        "source_text": data.get("source_text", ""),
        "translated_text": data.get("translated_text", ""),
        "confidence": _conf(data.get("confidence")),
    }


NORMALISERS: dict[str, Callable[[dict], dict[str, Any]]] = {
    "movie": _norm_movie,
    "transcribe": _norm_transcribe,
    "ocr": _norm_ocr,
    "describe": _norm_describe,
    "translate": _norm_translate,
}


def _conf(val) -> float:
    try:
        if val is None:
            return 0.0
        if isinstance(val, str) and val.strip().endswith("%"):
            return max(0.0, min(1.0, float(val.strip("% ")) / 100.0))
        f = float(val)
        if f > 1.0:
            f = f / 100.0
        return max(0.0, min(1.0, f))
    except (TypeError, ValueError):
        return 0.0


def normalise(mode: str, data: dict) -> dict[str, Any]:
    fn = NORMALISERS.get(mode, _norm_describe)
    try:
        return fn(data)
    except Exception:  # noqa: BLE001
        return {"confidence": 0.0, "raw": data}


def accepts(mode: str, media_kind: str) -> bool:
    """Whether a mode accepts a given media kind (video/audio/image)."""
    return media_kind in ACCEPTS.get(mode, set())


def media_kind_for(file_path) -> str:
    """Classify a downloaded file path into video/audio/image/other."""
    p = Path(file_path)
    ext = p.suffix.lower().lstrip(".")
    video = {"mp4", "mkv", "avi", "mov", "webm", "flv", "wmv", "m4v", "ts", "mpg", "mpeg", "3gp"}
    audio = {"mp3", "m4a", "aac", "flac", "ogg", "opus", "wav", "wma"}
    image = {"jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff", "heic"}
    if ext in video:
        return "video"
    if ext in audio:
        return "audio"
    if ext in image:
        return "image"
    return "other"


def render_result(mode: str, r: dict[str, Any]) -> str:
    conf = r.get("confidence")
    conf_s = f"{conf * 100:.0f}%" if conf is not None else "—"

    if mode == "movie":
        def cell(label, val):
            return f"  • *{label}:* {val if val not in (None,'','Unknown') else '—'}"
        return (
            "🎬 *AI Analysis Result*\n\n"
            f"{cell('Category', r.get('category'))}\n"
            f"{cell('Title', r.get('title'))}\n"
            f"{cell('Season', r.get('season'))}\n"
            f"{cell('Episode', r.get('episode'))}\n"
            f"{cell('Year', r.get('year'))}\n"
            f"{cell('Language', r.get('language'))}\n"
            f"{cell('Quality', r.get('quality'))}\n"
            f"{cell('Confidence', conf_s)}\n"
        )
    if mode == "transcribe":
        return (
            "🎵 *Transcription*\n\n"
            f"🌐 *Language:* {r.get('language') or '—'}\n"
            f"🎯 *Confidence:* {conf_s}\n\n"
            f"📝 *Text:*\n{r.get('text') or '_(no speech detected)_'}"
        )
    if mode == "ocr":
        return (
            "🔤 *Extracted Text*\n\n"
            f"🌐 *Language:* {r.get('language') or '—'}\n"
            f"🎯 *Confidence:* {conf_s}\n\n"
            f"📝 *Text:*\n```\n{r.get('text') or '_(no text found)_'}\n```"
        )
    if mode == "describe":
        return (
            "📝 *Scene Description*\n\n"
            f"*Summary:* {r.get('summary') or '—'}\n\n"
            f"*Details:*\n{r.get('details') or '—'}\n\n"
            f"🏷️ *Tags:* {r.get('tags') or '—'}\n"
            f"🎯 *Confidence:* {conf_s}"
        )
    if mode == "translate":
        return (
            "🌐 *Translation*\n\n"
            f"🔤 *Source language:* {r.get('source_language') or '—'}\n\n"
            f"📝 *Original:*\n```\n{r.get('source_text') or '_(no text)_'}\n```\n\n"
            f"✅ *Translation:*\n{r.get('translated_text') or '—'}\n\n"
            f"🎯 *Confidence:* {conf_s}"
        )
    return f"🤖 Result\n\n{r}"


# ===========================================================================
# Services — media_processor
# ===========================================================================

T = TypeVar("T")

# Global concurrency limiter for downloads/analyses.
_download_sem = asyncio.Semaphore(config.max_concurrent_downloads)


def download_slot():
    """Context manager acquiring one of the limited download slots."""
    return _download_sem


async def with_retries(
    func: Callable[..., Awaitable[T]],
    *args,
    retries: int | None = None,
    base_delay: float = 1.5,
    **kwargs,
) -> T:
    """Call ``func`` with retries and exponential back-off."""
    attempts = retries if retries is not None else config.max_retries
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == attempts:
                logger.warning("with_retries exhausted (%d): %s", attempt, exc)
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), 30.0)
            logger.info("retry %d/%d after %.1fs (%s)", attempt, attempts, delay, exc)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


def retrying(retries: int | None = None, base_delay: float = 1.5):
    """Decorator form of :func:`with_retries`."""

    def deco(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> T:
            return await with_retries(
                fn, *args, retries=retries, base_delay=base_delay, **kwargs
            )

        return wrapper

    return deco


async def send_file_back(
    chat_id: int,
    file_path: Path,
    caption: str,
    context: ContextTypes.DEFAULT_TYPE,
    progress_cb: Callable[[float, int, int], Awaitable[None]] | None = None,
) -> str:
    """Send a downloaded file back to the user via the Bot API.

    Returns a short status string: "ok" | "too_large" | "error:<msg>".
    """
    path = Path(file_path)
    if not path.exists():
        return f"error:file missing ({path.name})"
    size = path.stat().st_size

    if size > config.upload_limit_bytes:
        return "too_large"

    try:
        async with path.open("rb") as fh:
            await context.bot.send_document(
                chat_id=chat_id,
                document=fh,
                filename=path.name,
                caption=caption[:1024],
                read_timeout=config.download_timeout,
                write_timeout=config.download_timeout,
                connect_timeout=60,
                pool_timeout=60,
            )
        if progress_cb is not None:
            try:
                await progress_cb(100.0, size, size)
            except Exception:  # noqa: BLE001
                pass
        return "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("send_document failed: %s", exc)
        return f"error:{exc}"


async def notify(chat_id: int, context: ContextTypes.DEFAULT_TYPE,
                 text: str, reply_markup=None) -> None:
    """Best-effort notification sender (used when notifications are enabled)."""
    try:
        await context.bot.send_message(
            chat_id=chat_id, text=text, reply_markup=reply_markup,
            parse_mode="Markdown",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("notify failed: %s", exc)


async def safe_unlink(path) -> None:
    if path:
        await remove_path(path)


# ===========================================================================
# Services — downloader
# ===========================================================================

ProgressCB = Callable[[float, int, int, str], Awaitable[None]]

_FILE_URL_TEMPLATE = "https://api.telegram.org/file/bot{token}/{file_path}"


class DownloadError(Exception):
    pass


class FileTooLarge(DownloadError):
    pass


class _ProgressThrottle:
    """Calls the user callback at most every ``min_interval`` seconds OR
    every ``min_step`` percent, whichever comes first."""

    def __init__(self, cb: ProgressCB | None, *, min_interval: float = 1.0,
                 min_step: float = 4.0):
        self.cb = cb
        self.min_interval = min_interval
        self.min_step = min_step
        self._last_ts = 0.0
        self._last_pct = -100.0
        self._start_ts = 0.0
        self._received_at_start = 0

    def reset(self, received_at_start: int = 0) -> None:
        self._last_ts = 0.0
        self._last_pct = -100.0
        self._start_ts = time.monotonic()
        self._received_at_start = received_at_start

    async def _maybe_emit(self, received: int, total: int) -> None:
        if self.cb is None or total <= 0:
            return
        pct = received / total * 100.0
        now = time.monotonic()
        speed = ""
        delta = received - self._received_at_start
        elapsed = now - self._start_ts
        if elapsed > 0 and delta > 0:
            speed = _fmt_speed(delta / elapsed)
        due_time = (now - self._last_ts) >= self.min_interval
        due_step = (pct - self._last_pct) >= self.min_step
        if due_time or due_step or pct >= 100.0:
            self._last_ts = now
            self._last_pct = pct
            try:
                await self.cb(pct, received, total, speed)
            except Exception as exc:  # noqa: BLE001
                logger.debug("progress cb error (ignored): %s", exc)


def _fmt_speed(bps: float) -> str:
    for unit in ("B/s", "KB/s", "MB/s", "GB/s"):
        if bps < 1024 or unit == "GB/s":
            return f"{bps:.1f} {unit}"
        bps /= 1024
    return f"{bps:.1f} GB/s"


async def download_forwarded(
    message,
    *,
    progress: ProgressCB | None = None,
    bot,
) -> tuple[Path, str, str, int]:
    """Download media from a forwarded PTB message using the Bot API only.

    Returns ``(path, filename, media_type, size_bytes)``.
    """
    media_type = _ptb_media_type(message)
    file_id, declared_size, suggested_name = _ptb_file_meta(message, media_type)

    if not file_id:
        raise DownloadError("No downloadable media found in that message.")

    if declared_size and declared_size > config.download_limit_bytes:
        raise FileTooLarge(
            f"File is ~{declared_size // (1024 * 1024)} MB; the official Bot "
            f"API can download up to {config.max_file_size_mb} MB."
        )

    if suggested_name:
        suggested_name = safe_filename(suggested_name)
    else:
        ext = _ext_for_media_type(media_type)
        suggested_name = safe_filename(f"media_{message.message_id}{ext}")

    target = unique_path(config.downloads_dir, suggested_name)

    try:
        tg_file = await bot.get_file(file_id)
    except Exception as exc:  # noqa: BLE001
        raise DownloadError(f"getFile failed: {exc}") from exc

    remote_path = getattr(tg_file, "file_path", None)
    if not remote_path:
        raise DownloadError("Telegram returned no file_path for that media.")
    total_size = int(getattr(tg_file, "file_size", 0) or declared_size or 0)

    if config.local_mode and os.path.isabs(remote_path) and os.path.exists(remote_path):
        path = await _copy_local(Path(remote_path), target, total_size, progress)
        size = path.stat().st_size if path.exists() else 0
        return path, path.name, media_type, size

    url = _file_url(bot.token, remote_path)
    path = await _stream_to_disk(
        url, target, total_size=total_size, progress=progress, bot_token=bot.token
    )
    size = path.stat().st_size if path.exists() else 0
    return path, path.name, media_type, size


def _file_url(token: str, file_path: str) -> str:
    """Build the official Bot API file download URL for the active mode."""
    if config.local_mode:
        base = config.bot_api_file_url.rstrip("/")
        return f"{base}/file/bot{token}/{file_path}"
    return _FILE_URL_TEMPLATE.format(token=token, file_path=file_path)


async def _copy_local(src: Path, target: Path, total_size: int,
                      progress: ProgressCB | None) -> Path:
    """Stream-copy a locally-available file (local Bot API server mode)."""
    throttle = _ProgressThrottle(progress)
    throttle.reset()
    chunk = config.chunk_size_bytes
    received = 0
    async with aiofiles.open(src, "rb") as r, aiofiles.open(target, "wb") as w:
        while True:
            raw = await r.read(chunk)
            if not raw:
                break
            await w.write(raw)
            received += len(raw)
            await throttle._maybe_emit(received, total_size or received)
    return target


async def _stream_to_disk(
    url: str,
    target: Path,
    *,
    total_size: int,
    progress: ProgressCB | None,
    bot_token: str,
) -> Path:
    """Stream ``url`` to ``target`` in chunks, with resume + retries."""
    part_path = target.with_suffix(target.suffix + ".part")
    throttle = _ProgressThrottle(progress)

    async def _attempt() -> Path:
        existing = part_path.stat().st_size if part_path.exists() else 0
        throttle.reset(received_at_start=existing)

        headers: dict[str, str] = {}
        mode = "ab" if existing else "wb"
        if existing:
            headers["Range"] = f"bytes={existing}-"
            logger.info("Resuming %s from byte %d", target.name, existing)

        timeout = httpx.Timeout(
            connect=30.0,
            read=float(config.download_timeout),
            write=30.0,
            pool=60.0,
        )
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 416:
                    if part_path.exists() and total_size and \
                            part_path.stat().st_size >= total_size:
                        part_path.rename(target)
                        return target
                    part_path.unlink(missing_ok=True)
                    raise DownloadError("Range request not satisfiable.")

                if resp.status_code == 206:
                    mode_eff = "ab"
                    received = existing
                elif resp.status_code == 200:
                    mode_eff = "wb"
                    received = 0
                else:
                    raise DownloadError(
                        f"Telegram file server returned HTTP {resp.status_code}"
                    )

                content_length = resp.headers.get("Content-Length")
                if content_length and content_length.isdigit():
                    cl = int(content_length)
                    if mode_eff == "ab" and existing:
                        total = existing + cl
                    else:
                        total = cl
                else:
                    total = total_size or 0

                chunk = config.chunk_size_bytes
                async with aiofiles.open(part_path, mode_eff) as fh:
                    async for raw in resp.aiter_bytes(chunk):
                        await fh.write(raw)
                        received += len(raw)
                        if total:
                            await throttle._maybe_emit(min(received, total), total)

                final_size = part_path.stat().st_size
                if total and final_size < total:
                    raise DownloadError(
                        f"Incomplete download: {final_size}/{total} bytes"
                    )
                if total and final_size > total:
                    logger.warning("Downloaded %d > expected %d; restarting",
                                   final_size, total)
                    part_path.unlink(missing_ok=True)
                    raise DownloadError("Downloaded size exceeds expected.")

        part_path.rename(target)
        throttle.reset(received_at_start=0)
        await throttle._maybe_emit(total or target.stat().st_size,
                                   total or target.stat().st_size)
        return target

    try:
        return await with_retries(_attempt, retries=config.max_retries)
    except Exception:
        await remove_path(part_path)
        raise


def _ptb_media_type(message) -> str:
    if message.video:
        return "video"
    if message.animation:
        return "gif"
    if message.sticker:
        return "sticker"
    if message.video_note:
        return "video_note"
    if message.voice:
        return "voice"
    if message.audio:
        return "audio"
    if message.photo:
        return "photo"
    if message.document:
        return "document"
    return "file"


def _ptb_file_meta(message, media_type: str) -> tuple[Optional[str], int, Optional[str]]:
    """Extract (file_id, file_size, suggested_filename) from a PTB message."""
    candidates = (
        message.document, message.video, message.audio, message.animation,
        message.voice, message.video_note,
    )
    for cand in candidates:
        if cand is None:
            continue
        file_id = getattr(cand, "file_id", None)
        size = int(getattr(cand, "file_size", 0) or 0)
        name = getattr(cand, "file_name", None)
        if file_id:
            if not name:
                name = _fallback_name(message, media_type)
            return file_id, size, name
    if message.photo:
        largest = message.photo[-1]
        return largest.file_id, int(largest.file_size or 0), None
    if message.sticker:
        ext = ".webm" if (message.sticker.is_animated or message.sticker.is_video) else ".webp"
        return message.sticker.file_id, int(message.sticker.file_size or 0), \
            f"sticker_{message.sticker.file_unique_id}{ext}"
    return None, 0, None


def _fallback_name(message, media_type: str) -> str | None:
    if message.photo:
        return f"photo_{message.photo[-1].file_unique_id}.jpg"
    return None


def _ext_for_media_type(media_type: str) -> str:
    return {
        "video": ".mp4", "gif": ".mp4", "video_note": ".mp4",
        "voice": ".ogg", "audio": ".mp3", "photo": ".jpg",
        "sticker": ".webp", "document": ".bin", "file": ".bin",
    }.get(media_type, ".bin")


# ===========================================================================
# Services — frame_extractor
# ===========================================================================

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "ffprobe"


class FrameExtractionError(Exception):
    pass


async def extract_frames(
    video_path: Path,
    *,
    num_frames: Optional[int] = None,
    out_dir: Optional[Path] = None,
    fmt: str = "png",
    timeout: int = 300,
) -> list[Path]:
    """Extract ``num_frames`` evenly-spaced frames from *video_path*."""
    n = num_frames or config.num_frames
    n = max(1, min(n, 12))
    video_path = Path(video_path)
    if not video_path.exists():
        raise FrameExtractionError(f"Video not found: {video_path}")

    out_dir = Path(out_dir) if out_dir else config.frames_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = await _probe_duration(video_path)
    if not duration or duration < 0.1:
        raise FrameExtractionError("Could not determine video duration.")

    timestamps = _even_timestamps(duration, n)
    frames: list[Path] = []
    try:
        tasks = [
            _extract_at(video_path, ts, out_dir, i, fmt, timeout)
            for i, ts in enumerate(timestamps)
        ]
        for coro in asyncio.as_completed(tasks):
            frame = await coro
            if frame:
                frames.append(frame)
        frames.sort()
        if not frames:
            raise FrameExtractionError("No frames were extracted.")
        logger.info("Extracted %d/%d frames from %s", len(frames), n, video_path.name)
        return frames
    except Exception:
        for f in frames:
            await remove_path(f)
        raise


async def _extract_at(
    video_path: Path, ts: float, out_dir: Path, index: int,
    fmt: str, timeout: int,
) -> Optional[Path]:
    """Extract a single frame at timestamp ``ts`` (seconds)."""
    name = f"{video_path.stem}_f{index:02d}_{int(ts):06d}.{fmt}"
    out = unique_path(out_dir, name)
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-ss", f"{ts:.3f}", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "2",
        str(out),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.warning("ffmpeg timeout at ts=%.1f for %s", ts, video_path.name)
            await remove_path(out)
            return None
        if proc.returncode != 0:
            err = stderr.decode(errors="replace")[:200] if stderr else ""
            logger.warning("ffmpeg failed at ts=%.1f: %s", ts, err)
            await remove_path(out)
            return None
        if not out.exists() or out.stat().st_size == 0:
            await remove_path(out)
            return None
        return out
    except FileNotFoundError as exc:
        raise FrameExtractionError("ffmpeg is not installed") from exc


async def _probe_duration(video_path: Path) -> float:
    """Get duration in seconds via ffprobe."""
    cmd = [
        _FFPROBE, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
    except (FileNotFoundError, asyncio.TimeoutError) as exc:
        logger.warning("ffprobe failed: %s — falling back to ffmpeg parse", exc)
        return await _duration_via_ffmpeg(video_path)
    text = stdout.decode(errors="replace").strip()
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return float(m.group()) if m else 0.0


async def _duration_via_ffmpeg(video_path: Path) -> float:
    """Fallback duration detection by parsing ffmpeg stderr."""
    cmd = [_FFMPEG, "-i", str(video_path)]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    except (FileNotFoundError, asyncio.TimeoutError):
        return 0.0
    text = stderr.decode(errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not m:
        return 0.0
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + s


def _even_timestamps(duration: float, n: int) -> list[float]:
    """Evenly spaced timestamps, avoiding the very first/last frames."""
    if n == 1:
        return [duration / 2]
    step = duration / (n + 1)
    return [round(step * (i + 1), 3) for i in range(n)]


# ===========================================================================
# Services — ai_analyzer
# ===========================================================================

_clients: dict[str | None, Any] = {}
_client_errors: dict[str | None, str] = {}


def _get_client(api_key: str | None = None):
    """Return a Gemini client for the given key (cached)."""
    key = api_key or config.gemini_api_key or None
    cache_key = key
    if cache_key in _clients:
        return _clients[cache_key]
    if cache_key in _client_errors:
        raise RuntimeError(_client_errors[cache_key])
    if not key:
        _client_errors[cache_key] = (
            "No Gemini API key configured. Set GEMINI_API_KEY or add your "
            "own key in Settings."
        )
        raise RuntimeError(_client_errors[cache_key])
    try:
        from google import genai  # type: ignore
        client = genai.Client(api_key=key)
        _clients[cache_key] = client
        logger.info("Gemini client initialised (key=%s).",
                    "user" if api_key else "global")
        return client
    except Exception as exc:  # noqa: BLE001
        _client_errors[cache_key] = f"Gemini init failed: {exc}"
        raise RuntimeError(_client_errors[cache_key]) from exc


async def analyze_frames(
    frame_paths: Sequence[Path],
    *,
    model: str | None = None,
    mode: str = "movie",
    user_api_key: str | None = None,
    target_language: str | None = None,
) -> dict[str, Any]:
    """Send frames to Gemini in the given mode and return a normalised result."""
    model = model or config.default_ai_model
    if not frame_paths:
        raise ValueError("No frames to analyse.")

    client = _get_client(user_api_key)
    prompt = ai_modes.build_prompt(mode, target_language=target_language)

    parts = await _build_parts(frame_paths, mode)
    parts.append(prompt)

    async def _call():
        from google.genai import types  # type: ignore
        return await _run_in_thread(
            client.models.generate_content,
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=2000,
            ),
        )

    try:
        response = await with_retries(_call, retries=config.max_retries)
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini request failed: %s", exc)
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    text = _extract_text(response)
    data = _parse_json(text)
    return ai_modes.normalise(mode, data)


async def analyze_audio(
    audio_path: Path,
    *,
    model: str | None = None,
    mode: str = "transcribe",
    user_api_key: str | None = None,
    target_language: str | None = None,
) -> dict[str, Any]:
    """Send an audio file to Gemini (for transcription mode)."""
    model = model or config.default_ai_model
    client = _get_client(user_api_key)
    prompt = ai_modes.build_prompt(mode, target_language=target_language)

    parts = await _build_audio_parts(audio_path)
    parts.append(prompt)

    async def _call():
        from google.genai import types  # type: ignore
        return await _run_in_thread(
            client.models.generate_content,
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=2000,
            ),
        )

    response = await with_retries(_call, retries=config.max_retries)
    text = _extract_text(response)
    data = _parse_json(text)
    return ai_modes.normalise(mode, data)


async def analyze_images(
    image_paths: Sequence[Path],
    *,
    model: str | None = None,
    mode: str = "ocr",
    user_api_key: str | None = None,
    target_language: str | None = None,
) -> dict[str, Any]:
    """Send image files to Gemini (for ocr/describe/translate modes)."""
    model = model or config.default_ai_model
    client = _get_client(user_api_key)
    prompt = ai_modes.build_prompt(mode, target_language=target_language)

    parts = await _build_parts(image_paths, mode)
    parts.append(prompt)

    async def _call():
        from google.genai import types  # type: ignore
        return await _run_in_thread(
            client.models.generate_content,
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=2000,
            ),
        )

    response = await with_retries(_call, retries=config.max_retries)
    text = _extract_text(response)
    data = _parse_json(text)
    return ai_modes.normalise(mode, data)


async def _build_parts(paths: Sequence[Path], mode: str) -> list[Any]:
    from google.genai import types  # type: ignore

    mime_by_ext = {".png": "image/png", ".jpg": "image/jpeg",
                   ".jpeg": "image/jpeg", ".webp": "image/webp"}
    parts: list[Any] = []
    for fp in paths:
        fp = Path(fp)
        if not fp.exists():
            continue
        data = await _read_bytes(fp)
        mime = mime_by_ext.get(fp.suffix.lower(), "image/png")
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    if not parts:
        raise ValueError("No readable frame/image files.")
    return parts


async def _build_audio_parts(audio_path: Path) -> list[Any]:
    from google.genai import types  # type: ignore

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise ValueError("Audio file not found.")
    data = await _read_bytes(audio_path)
    ext = audio_path.suffix.lower()
    mime = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".ogg": "audio/ogg",
            ".opus": "audio/opus", ".wav": "audio/wav", ".flac": "audio/flac",
            ".aac": "audio/aac"}.get(ext, "audio/mpeg")
    return [types.Part.from_bytes(data=data, mime_type=mime)]


async def _read_bytes(path: Path) -> bytes:
    async with aiofiles.open(path, "rb") as fh:
        return await fh.read()


async def _run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _extract_text(response) -> str:
    try:
        text = getattr(response, "text", None)
        if text:
            return text
    except Exception:  # noqa: BLE001
        pass
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        pts = getattr(content, "parts", None) or []
        for p in pts:
            t = getattr(p, "text", None)
            if t:
                return t
    return ""


def _parse_json(text: str) -> dict:
    if not text:
        return {}
    cleaned = _strip_codefences(text).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _strip_codefences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


async def cleanup_frames(frame_paths: Sequence[Path]) -> None:
    for fp in frame_paths:
        await remove_path(fp)


# ===========================================================================
# Services — media_tools
# ===========================================================================


class ToolError(Exception):
    pass


async def extract_audio(video_path: Path, out_dir: Path,
                        bitrate: str = "192k") -> Path:
    video_path = Path(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = unique_path(out_dir, f"{video_path.stem}.mp3")
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame", "-b:a", bitrate,
        str(out),
    ]
    await _run_ffmpeg(cmd, out, label="extract_audio")
    return out


async def extract_thumbnail(video_path: Path, out_dir: Path,
                            ts: float = 1.0) -> Path:
    video_path = Path(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = unique_path(out_dir, f"{video_path.stem}_thumb.jpg")
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-ss", f"{max(0.0, ts):.3f}", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "2",
        str(out),
    ]
    await _run_ffmpeg(cmd, out, label="extract_thumbnail")
    return out


async def media_info(path: Path) -> dict:
    path = Path(path)
    cmd = [
        _FFPROBE, "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    except (FileNotFoundError, asyncio.TimeoutError) as exc:
        raise ToolError(f"ffprobe unavailable: {exc}") from exc
    if proc.returncode != 0:
        raise ToolError(
            f"ffprobe failed: {stderr.decode(errors='replace')[:200]}"
        )
    try:
        data = json.loads(stdout.decode(errors="replace"))
    except json.JSONDecodeError as exc:
        raise ToolError(f"ffprobe returned non-JSON: {exc}") from exc
    return _normalise_info(data)


def _normalise_info(data: dict) -> dict:
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})

    def g(d, *keys, default=None):
        for k in keys:
            if k in d and d[k] not in (None, ""):
                return d[k]
        return default

    out = {
        "format": g(fmt, "format_long_name", "format_name"),
        "duration": _fmt_duration(g(fmt, "duration")),
        "size_bytes": g(fmt, "size"),
        "bit_rate": g(fmt, "bit_rate"),
        "video_codec": g(v, "codec_long_name", "codec_name"),
        "width": g(v, "width"),
        "height": g(v, "height"),
        "fps": _parse_fps(g(v, "r_frame_rate")),
        "audio_codec": g(a, "codec_long_name", "codec_name"),
        "channels": g(a, "channels"),
        "sample_rate": g(a, "sample_rate"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _fmt_duration(raw) -> str:
    try:
        secs = float(raw)
    except (TypeError, ValueError):
        return ""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    if h:
        return f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{s:05.2f}"


def _parse_fps(rate) -> str:
    if not rate or "/" not in rate:
        return str(rate) if rate else ""
    try:
        num, den = rate.split("/")
        den = float(den)
        if den == 0:
            return ""
        return f"{float(num) / den:.2f}"
    except (ValueError, ZeroDivisionError):
        return ""


async def compress_video(video_path: Path, out_dir: Path, *,
                         crf: int = 28, preset: str = "veryfast",
                         scale: Optional[str] = None) -> Path:
    video_path = Path(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = unique_path(out_dir, f"{video_path.stem}_compressed.mp4")
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-c:a", "aac", "-b:a", "128k",
    ]
    vf = []
    if scale:
        vf.append(f"scale={scale}")
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-movflags", "+faststart", str(out)]
    await _run_ffmpeg(cmd, out, label="compress_video")
    return out


_PILLO_FORMATS = {
    "png": "PNG", "jpg": "JPEG", "jpeg": "JPEG",
    "webp": "WEBP", "bmp": "BMP",
}


async def convert_image(img_path: Path, out_dir: Path, fmt: str) -> Path:
    fmt = (fmt or "").lower().lstrip(".")
    if fmt not in _PILLO_FORMATS:
        raise ToolError(
            f"Unsupported target format '{fmt}'. Use one of: "
            + ", ".join(_PILLO_FORMATS)
        )
    pil_fmt = _PILLO_FORMATS[fmt]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = unique_path(out_dir, f"{img_path.stem}.{fmt}")

    def _do() -> None:
        with Image.open(img_path) as im:
            if pil_fmt == "JPEG" and im.mode in ("RGBA", "P", "LA"):
                im = im.convert("RGB")
            im.save(out, format=pil_fmt)

    try:
        await asyncio.to_thread(_do)
    except Exception as exc:  # noqa: BLE001
        await remove_path(out)
        raise ToolError(f"Image conversion failed: {exc}") from exc
    return out


async def _run_ffmpeg(cmd: list[str], out_path: Path, *,
                      label: str, timeout: int = 600) -> None:
    logger.info("%s: %s", label, " ".join(cmd[:3] + ["…", str(out_path)]))
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            await remove_path(out_path)
            raise ToolError(f"{label} timed out after {timeout}s")
        if proc.returncode != 0:
            err = stderr.decode(errors="replace")[:300] if stderr else ""
            await remove_path(out_path)
            raise ToolError(f"{label} failed: {err}")
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise ToolError(f"{label} produced no output")
    except FileNotFoundError as exc:
        await remove_path(out_path)
        raise ToolError("ffmpeg is not installed") from exc


# ===========================================================================
# Services — qr_generator
# ===========================================================================


class QRError(Exception):
    pass


async def generate_qr(text: str, out_dir: Path, *,
                      box_size: int = 10, border: int = 2) -> Path:
    """Render ``text`` as a high-resolution PNG qr code. Returns its path."""
    if not text or not text.strip():
        raise QRError("Cannot encode empty text.")
    if len(text) > 2900:
        raise QRError("Text too long for a QR code (max ~2900 chars).")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = unique_path(out_dir, "qr.png")

    def _do() -> None:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=box_size,
            border=border,
        )
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(out, format="PNG")

    try:
        await asyncio.to_thread(_do)
    except Exception as exc:  # noqa: BLE001
        raise QRError(f"QR generation failed: {exc}") from exc
    logger.info("QR generated for %d-char input -> %s", len(text), out.name)
    return out


# ===========================================================================
# Services — backup
# ===========================================================================


async def export_user_json(user_id: int) -> Path:
    """Export the user's data to a JSON file and return its path."""
    data = await repo.export_user_data(user_id)
    data["exported_at"] = datetime.now(timezone.utc).isoformat()
    data["bot_version"] = "mediagrab-ai-bot"
    out_dir = config.frames_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = safe_filename(f"mediagrab_backup_{user_id}.json")
    target = out_dir / name
    i = 1
    while target.exists():
        target = out_dir / f"mediagrab_backup_{user_id}_{i}.json"
        i += 1
    text = json.dumps(data, indent=2, default=str, ensure_ascii=False)
    async with aiofiles.open(target, "w", encoding="utf-8") as fh:
        await fh.write(text)
    logger.info("Exported backup for user %s -> %s (%d bytes)",
                user_id, target.name, len(text))
    return target


async def parse_backup_file(path: Path) -> dict:
    """Read a backup JSON file into a dict."""
    async with aiofiles.open(path, "r", encoding="utf-8") as fh:
        text = await fh.read()
    return json.loads(text)


async def restore_from_dict(user_id: int, data: dict) -> dict[str, int]:
    """Restore a user's settings from a backup dict."""
    settings = data.get("settings") or {}
    await repo.restore_user_settings(user_id, settings)
    return {"settings": len(settings)}


async def export_global_json() -> Path:
    """Admin-only: export aggregate global stats (no per-user PII)."""
    stats = await repo.global_stats()
    data = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "global_stats": stats,
    }
    out_dir = config.frames_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "mediagrab_global_stats.json"
    text = json.dumps(data, indent=2, default=str, ensure_ascii=False)
    async with aiofiles.open(target, "w", encoding="utf-8") as fh:
        await fh.write(text)
    return target


# ===========================================================================
# Services — link_parser (Task ID 5)
# ===========================================================================

# Input kind constants.
KIND_MESSAGE_LINK = "message_link"        # link with a message id
KIND_CHANNEL_LINK = "channel_link"        # link without a message id
KIND_USERNAME = "username"                # @username or bare username
KIND_CHAT_ID = "chat_id"                  # raw numeric id

# Source visibility (affects which Bot API calls can succeed).
VIS_PUBLIC = "public"      # resolvable by anyone via @username
VIS_PRIVATE = "private"    # /c/<internal_id>/ form; bot must be a member


@dataclass
class ParsedInput:
    """Structured result of parsing a user's download input."""

    kind: str = "unknown"                 # one of KIND_*
    visibility: str = ""                  # VIS_PUBLIC | VIS_PRIVATE | ""
    username: Optional[str] = None        # without leading @
    chat_id: Optional[int] = None         # resolved numeric id (private links)
    message_id: Optional[int] = None      # when present
    thread_id: Optional[int] = None       # topic thread id, if any
    raw: str = ""                         # original input
    error: str = ""                       # populated when unusable

    @property
    def ok(self) -> bool:
        return self.kind != "unknown" and not self.error

    def summary(self) -> str:
        parts = [f"kind={self.kind}"]
        if self.visibility:
            parts.append(f"vis={self.visibility}")
        if self.username:
            parts.append(f"user=@{self.username}")
        if self.chat_id is not None:
            parts.append(f"chat_id={self.chat_id}")
        if self.message_id is not None:
            parts.append(f"msg_id={self.message_id}")
        if self.thread_id is not None:
            parts.append(f"thread={self.thread_id}")
        if self.error:
            parts.append(f"error={self.error}")
        return " | ".join(parts)


# Regex patterns ----------------------------------------------------------

# Username rules: 5+ chars, alnum + underscore, must start with a letter.
_USERNAME_PATTERN = r"([A-Za-z][A-Za-z0-9_]{4,})"
_MSGID_PATTERN = r"(\d+)"

# Match t.me/c/<internal_id>/<msgid>  (private supergroup/channel)
_PRIVATE_LINK_RE = re.compile(
    r"https?://t(?:elegram)?\.me/c/(\d+)/(\d+)(?:/(\d+))?",
    re.IGNORECASE,
)
# Match t.me/<username>/<msgid>[/thread]  (public)
_PUBLIC_MSG_LINK_RE = re.compile(
    r"https?://t(?:elegram)?\.me/" + _USERNAME_PATTERN + r"/" + _MSGID_PATTERN + r"(?:/" + _MSGID_PATTERN + r")?",
    re.IGNORECASE,
)
# Match t.me/<username>  (channel link, no message id)
_PUBLIC_CHANNEL_LINK_RE = re.compile(
    r"https?://t(?:elegram)?\.me/" + _USERNAME_PATTERN + r"/?$",
    re.IGNORECASE,
)
# Match t.me/c/<internal_id>  (private channel link, no message id)
_PRIVATE_CHANNEL_LINK_RE = re.compile(
    r"https?://t(?:elegram)?\.me/c/(\d+)/?$",
    re.IGNORECASE,
)
# Match t.me/joinchat/<hash> (invite link — not a message link)
_INVITE_LINK_RE = re.compile(
    r"https?://t(?:elegram)?\.me/(?:joinchat/|\+)([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
# Bare @username
_AT_USERNAME_RE = re.compile(r"^@([A-Za-z][A-Za-z0-9_]{4,})$")
# Bare username (no @)
_BARE_USERNAME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]{4,})$")
# Raw chat id (-100... or plain number)
_CHAT_ID_RE = re.compile(r"^(-?\d{6,})$")


def parse_input(raw: str) -> ParsedInput:
    """Parse arbitrary user input into a structured :class:`ParsedInput`.

    Never raises — returns a ParsedInput with ``error`` set on failure.
    """
    if not raw:
        return ParsedInput(error="Empty input.", raw=raw or "")
    text = raw.strip()
    if not text:
        return ParsedInput(error="Empty input.", raw=raw)

    # Strip surrounding markdown/whitespace artefacts users sometimes paste.
    text = text.strip("`<>()")

    # 1) Private message link: t.me/c/<id>/<msgid>
    m = _parse_private_link(text)
    if m:
        return m

    # 2) Public message link: t.me/<username>/<msgid>
    m = _PUBLIC_MSG_LINK_RE.search(text)
    if m:
        username = m.group(1)
        msg_id = int(m.group(2))
        thread = int(m.group(3)) if m.group(3) else None
        return ParsedInput(
            kind=KIND_MESSAGE_LINK,
            visibility=VIS_PUBLIC,
            username=username,
            message_id=msg_id,
            thread_id=thread,
            raw=text,
        )

    # 3) Private channel link (no msg id): t.me/c/<id>
    m = _PRIVATE_CHANNEL_LINK_RE.search(text)
    if m:
        internal_id = int(m.group(1))
        return ParsedInput(
            kind=KIND_CHANNEL_LINK,
            visibility=VIS_PRIVATE,
            chat_id=_to_supergroup_id(internal_id),
            raw=text,
        )

    # 4) Public channel link (no msg id): t.me/<username>
    m = _PUBLIC_CHANNEL_LINK_RE.search(text)
    if m:
        return ParsedInput(
            kind=KIND_CHANNEL_LINK,
            visibility=VIS_PUBLIC,
            username=m.group(1),
            raw=text,
        )

    # 5) Invite link — cannot download from these.
    if _INVITE_LINK_RE.search(text):
        return ParsedInput(
            error="Invite links (t.me/+… or t.me/joinchat/…) can't be used to "
                  "download media. Send a message link or a public @username.",
            raw=text,
        )

    # 6) Bare @username
    m = _AT_USERNAME_RE.match(text)
    if m:
        return ParsedInput(
            kind=KIND_USERNAME,
            visibility=VIS_PUBLIC,
            username=m.group(1),
            raw=text,
        )

    # 7) Bare username (no @) — be lenient but validate length/charset.
    m = _BARE_USERNAME_RE.match(text)
    if m:
        return ParsedInput(
            kind=KIND_USERNAME,
            visibility=VIS_PUBLIC,
            username=m.group(1),
            raw=text,
        )

    # 8) Raw chat id
    m = _CHAT_ID_RE.match(text)
    if m:
        cid = int(m.group(1))
        return ParsedInput(
            kind=KIND_CHAT_ID,
            chat_id=cid,
            raw=text,
        )

    # 9) Anything else — try to salvage a t.me link with weird extra path.
    if "t.me/" in text.lower() or "telegram.me/" in text.lower():
        return ParsedInput(
            error="That looks like a Telegram link, but I couldn't parse it. "
                  "Supported: t.me/<username>/<msgid>, t.me/c/<id>/<msgid>, "
                  "or @username.",
            raw=text,
        )

    return ParsedInput(
        error="Unrecognised input. Send a Telegram message link "
              "(https://t.me/…), a @username, or a channel link.",
        raw=text,
    )


def _parse_private_link(text: str) -> Optional[ParsedInput]:
    """Handle t.me/c/<id>/<msgid> (private message link)."""
    m = _PRIVATE_LINK_RE.search(text)
    if not m:
        return None
    internal_id = int(m.group(1))
    msg_id = int(m.group(2))
    thread = int(m.group(3)) if m.group(3) else None
    return ParsedInput(
        kind=KIND_MESSAGE_LINK,
        visibility=VIS_PRIVATE,
        chat_id=_to_supergroup_id(internal_id),
        message_id=msg_id,
        thread_id=thread,
        raw=text,
    )


def _to_supergroup_id(internal_id: int) -> int:
    """Convert the internal id from t.me/c/<id> into a Bot API chat id.

    Telegram's Bot API uses -100 prefixed to the internal id for supergroups
    and channels. ``t.me/c/1234567890`` corresponds to chat id
    ``-1001234567890``.
    """
    return -1000000000000 - internal_id


def from_chat_reference(parsed: ParsedInput) -> str | int | None:
    """Return the value to pass as ``from_chat_id`` to copyMessage/forwardMessage.

    For public inputs this is the ``@username`` string; for private inputs
    it's the numeric ``-100…`` chat id. Returns None if not resolvable.
    """
    if parsed.username:
        return f"@{parsed.username}"
    if parsed.chat_id is not None:
        return parsed.chat_id
    return None


def describe_input(parsed: ParsedInput) -> str:
    """Human-friendly one-line description for status messages."""
    if not parsed.ok:
        return f"invalid input ({parsed.error})"
    if parsed.kind == KIND_MESSAGE_LINK:
        loc = (f"@{parsed.username}" if parsed.username
               else str(parsed.chat_id))
        return f"message #{parsed.message_id} from {loc}"
    if parsed.kind == KIND_CHANNEL_LINK:
        loc = (f"@{parsed.username}" if parsed.username
               else f"private chat {parsed.chat_id}")
        return f"channel {loc} (no message id yet)"
    if parsed.kind == KIND_USERNAME:
        return f"public chat @{parsed.username}"
    if parsed.kind == KIND_CHAT_ID:
        return f"chat id {parsed.chat_id}"
    return "unknown input"


# ===========================================================================
# UI — keyboards
# ===========================================================================

MAIN = "m:main"
BACK_MAIN = "b:main"


def _back(destination: str = "main") -> InlineKeyboardButton:
    return InlineKeyboardButton("🔙 Back", callback_data=f"b:{destination}")


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📥 Download", callback_data="m:dl"),
            InlineKeyboardButton("🎬 AI Analyze", callback_data="m:ai"),
        ],
        [
            InlineKeyboardButton("🔍 Inspect Chat", callback_data="m:ins"),
            InlineKeyboardButton("🧰 Media Toolbox", callback_data="m:tb"),
        ],
        [
            InlineKeyboardButton("⭐ Library", callback_data="m:lib"),
            InlineKeyboardButton("📊 Stats", callback_data="m:st"),
        ],
        [
            InlineKeyboardButton("🔳 QR Code", callback_data="m:qr"),
            InlineKeyboardButton("📦 Batch / Album", callback_data="m:batch"),
        ],
        [
            InlineKeyboardButton("⏰ Scheduled", callback_data="m:sched"),
            InlineKeyboardButton("💾 Backup", callback_data="m:bk"),
        ],
        [
            InlineKeyboardButton("📜 History", callback_data="m:hist"),
            InlineKeyboardButton("⚙️ Settings", callback_data="m:set"),
        ],
        [
            InlineKeyboardButton("ℹ️ Help", callback_data="m:help"),
            InlineKeyboardButton("🛡️ Admin", callback_data="m:admin"),
        ],
    ])


def admin_only_menu(is_admin: bool = False) -> InlineKeyboardMarkup:
    """Main menu variant — shows Admin button only to admins."""
    if is_admin:
        return main_menu()
    return main_menu()


def download_menu() -> InlineKeyboardMarkup:
    """Download menu — forward media OR download by link/@username."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("↩️ Forward media to me", callback_data="dl:fwd")],
        [InlineKeyboardButton("🔗 Download by link or @username", callback_data="dl:link")],
        [_back("main")],
    ])


def analyze_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎬 Forward media to analyze", callback_data="ai:fwd")],
        [InlineKeyboardButton("🧠 Choose AI Mode", callback_data="ai:modes")],
        [_back("main")],
    ])


def ai_modes_menu(current_mode: str = "movie") -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for mode in MODE_ORDER:
        label = MODE_LABELS.get(mode, mode)
        marker = "✅ " if mode == current_mode else "  "
        rows.append([InlineKeyboardButton(
            f"{marker}{label}", callback_data=f"ai:mode:{mode}"
        )])
    rows.append([_back("ai")])
    return InlineKeyboardMarkup(rows)


def inspector_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔍 Inspect a @username or link", callback_data="ins:fwd")],
        [InlineKeyboardButton("🕘 Recent Inspections", callback_data="ins:recent")],
        [_back("main")],
    ])


def toolbox_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎵 Extract Audio (MP3)", callback_data="tb:audio")],
        [InlineKeyboardButton("🖼️ Extract Thumbnail", callback_data="tb:thumb")],
        [InlineKeyboardButton("📋 Media Info", callback_data="tb:info")],
        [InlineKeyboardButton("🎬 Compress Video", callback_data="tb:compress")],
        [InlineKeyboardButton("🔄 Convert Image", callback_data="tb:imgconv")],
        [_back("main")],
    ])


def library_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📚 Browse All", callback_data="lib:browse"),
            InlineKeyboardButton("🔍 Search", callback_data="lib:search"),
        ],
        [
            InlineKeyboardButton("🎬 Videos", callback_data="lib:t:video"),
            InlineKeyboardButton("🎵 Audio", callback_data="lib:t:audio"),
        ],
        [
            InlineKeyboardButton("🖼️ Images", callback_data="lib:t:photo"),
            InlineKeyboardButton("📄 Documents", callback_data="lib:t:document"),
        ],
        [InlineKeyboardButton("🧹 Clear Library", callback_data="lib:clear")],
        [_back("main")],
    ])


def library_entry_keyboard(entry_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🗑️ Remove", callback_data=f"lib:del:{entry_id}")],
        [_back("lib")],
    ])


def stats_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("👤 My Stats", callback_data="st:me"),
            InlineKeyboardButton("🌍 Global", callback_data="st:global"),
        ],
        [_back("main")],
    ])


def qr_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔳 Generate a QR Code", callback_data="qr:make")],
        [_back("main")],
    ])


def batch_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(
            "📦 Forward an album (media group)", callback_data="batch:fwd"
        )],
        [_back("main")],
    ])


def scheduled_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 View scheduled tasks", callback_data="sched:list")],
        [_back("main")],
    ])


def backup_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💾 Export my data (JSON)", callback_data="bk:export")],
        [InlineKeyboardButton("📥 Import settings from JSON", callback_data="bk:import")],
        [_back("main")],
    ])


def admin_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("👥 List users", callback_data="adm:users")],
        [InlineKeyboardButton("📣 Broadcast message", callback_data="adm:bcast")],
        [InlineKeyboardButton("📊 Global stats", callback_data="adm:stats")],
        [InlineKeyboardButton("💾 Export global stats", callback_data="adm:export")],
        [InlineKeyboardButton("🛰️ MTProto Backend", callback_data="adm:mtp")],
        [InlineKeyboardButton("🎙 VC Control", callback_data="adm:vc")],
        [_back("main")],
    ])


def mtproto_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("▶️ Start", callback_data="mtp:start"),
            InlineKeyboardButton("⏹️ Stop", callback_data="mtp:stop"),
        ],
        [
            InlineKeyboardButton("🔄 Restart", callback_data="mtp:restart"),
            InlineKeyboardButton("📊 Refresh", callback_data="mtp:status"),
        ],
        [InlineKeyboardButton("📸 Screenshot", callback_data="mtp:screenshot")],
        [InlineKeyboardButton("📥 MTProto Download", callback_data="mtp:download")],
        [_back("admin")],
    ])


def history_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📥 Recent Downloads", callback_data="h:dl"),
            InlineKeyboardButton("🤖 Recent AI Analyses", callback_data="h:ai"),
        ],
        [
            InlineKeyboardButton("🧹 Clear Downloads", callback_data="h:clr:dl"),
            InlineKeyboardButton("🧹 Clear AI", callback_data="h:clr:ai"),
        ],
        [InlineKeyboardButton("🧹 Clear All", callback_data="h:clr:all")],
        [_back("main")],
    ])


def settings_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🎯 Quality", callback_data="s:q"),
            InlineKeyboardButton("🧠 AI Model", callback_data="s:m"),
        ],
        [
            InlineKeyboardButton("🧠 AI Mode", callback_data="s:am"),
            InlineKeyboardButton("🌐 Language", callback_data="s:l"),
        ],
        [
            InlineKeyboardButton("🗑️ Auto Delete", callback_data="s:ad"),
            InlineKeyboardButton("🔔 Notifications", callback_data="s:n"),
        ],
        [InlineKeyboardButton("🔑 Gemini API Key", callback_data="s:key")],
        [_back("main")],
    ])


def help_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📥 How to Download Media", callback_data="hp:dl")],
        [InlineKeyboardButton("🤖 How AI Recognition Works", callback_data="hp:ai")],
        [InlineKeyboardButton("🔍 Inspect & Toolbox", callback_data="hp:feat")],
        [InlineKeyboardButton("📄 Supported Formats", callback_data="hp:fmt")],
        [InlineKeyboardButton("❓ Frequently Asked Questions", callback_data="hp:faq")],
        [_back("main")],
    ])


def back_only(destination: str = "main") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back(destination)]])


def cancel_back(destination: str = "main") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("❌ Cancel", callback_data=f"b:{destination}")],
    ])


def settings_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("set")]])


def help_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("help")]])


def history_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("hist")]])


def library_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("lib")]])


def inspector_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("ins")]])


def toolbox_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("tb")]])


def stats_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("st")]])


def qr_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("qr")]])


def download_done_keyboard() -> InlineKeyboardMarkup:
    """Shown after a download completes — offer to bookmark to Library."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⭐ Save to Library", callback_data="lib:savelast")],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="b:main")],
    ])


def batch_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("batch")]])


def scheduled_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("sched")]])


def backup_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("bk")]])


def admin_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("admin")]])


def ai_modes_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("ai")]])


def mtproto_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("mtproto")]])


# ---------------------------------------------------------------------------
# VC Tour control
# ---------------------------------------------------------------------------

def vc_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("▶️ Start VC Tour", callback_data="vc:start"),
            InlineKeyboardButton("⏸ Pause Tour", callback_data="vc:pause"),
        ],
        [
            InlineKeyboardButton("▶️ Resume Tour", callback_data="vc:resume"),
            InlineKeyboardButton("⏹ Stop Tour", callback_data="vc:stop"),
        ],
        [
            InlineKeyboardButton("📍 Current VC", callback_data="vc:current"),
            InlineKeyboardButton("📊 Tour Status", callback_data="vc:status"),
        ],
        [
            InlineKeyboardButton("📜 VC History", callback_data="vc:hist:0"),
            InlineKeyboardButton("🔄 Refresh Groups", callback_data="vc:refresh"),
        ],
        [
            InlineKeyboardButton("🎯 Manual VC Control", callback_data="vc:manual"),
            InlineKeyboardButton("⚙️ VC Settings", callback_data="vc:settings"),
        ],
        [_back("admin")],
    ])


def vc_manual_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎙 Join Target VC", callback_data="vc:jointarget")],
        [InlineKeyboardButton("🚪 Leave Current VC", callback_data="vc:leave")],
        [
            InlineKeyboardButton("⏱ Stay 5 Min", callback_data="vc:stay5"),
            InlineKeyboardButton("⏱ Custom Duration", callback_data="vc:staycustom"),
        ],
        [
            InlineKeyboardButton("🔎 Check Target VC", callback_data="vc:checktarget"),
            InlineKeyboardButton("📍 Current VC", callback_data="vc:current"),
        ],
        [_back("vc")],
    ])


def vc_settings_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("vc")]])


def vc_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("vc")]])


def vcmanual_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[_back("vcmanual")]])


# ===========================================================================
# UI — messages
# ===========================================================================

WELCOME = (
    "👋 *Welcome to MediaGrab AI Bot*\n\n"
    "Your all-in-one assistant for:\n"
    "📥 Downloading Telegram media\n"
    "🎬 AI-powered video recognition\n"
    "📜 History of your activity\n\n"
    "Tap a button below to begin — no commands needed."
)

MAIN_MENU_TEXT = (
    "*🏠 Main Menu*\n\n"
    "Choose what you'd like to do:"
)

DOWNLOAD_MENU_TEXT = (
    "*📥 Download Media*\n\n"
    "Two ways to download:\n\n"
    "• *↩️ Forward media to me* — forward any message with media\n"
    "• *🔗 Download by link or @username* — paste a t.me link or send "
    "a @channelusername\n\n"
    f"_ℹ️ Uses the official Bot API — files up to {config.max_file_size_mb} MB._"
)

ANALYZE_MENU_TEXT = (
    "*🎬 AI Analyze*\n\n"
    "Forward me a *video* and I'll:\n"
    "1️⃣ Extract several frames\n"
    "2️⃣ Send them to Google Gemini\n"
    "3️⃣ Return the detected title, season, episode, year & more\n\n"
    f"_ℹ️ Video file up to {config.max_file_size_mb} MB (Bot API limit)._"
)

HISTORY_MENU_TEXT = (
    "*📜 History*\n\n"
    "View or clear your recent activity."
)

SETTINGS_MENU_TEXT = (
    "*⚙️ Settings*\n\n"
    "Tap any option to cycle through its values."
)

HELP_MENU_TEXT = (
    "*ℹ️ Help*\n\n"
    "Pick a topic to learn more."
)

QUALITIES = ("Original", "1080p", "720p", "480p")
AI_MODELS = ("gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro")
LANGUAGES = ("English", "हिन्दी", "Español", "français", "العربية", "中文", "português")


def settings_text(s: dict) -> str:
    def on_off(v) -> str:
        return "✅ On" if v else "⚪️ Off"

    ai_mode = s.get("ai_mode", "movie")
    has_key = "✅ Set" if s.get("gemini_api_key") else "⚪️ Using shared key"
    mode_label = MODE_LABELS.get(ai_mode, ai_mode)
    return (
        "*⚙️ Settings*\n\n"
        f"🎯 *Preferred Quality:* {s.get('preferred_quality')}\n"
        f"🧠 *AI Model:* `{s.get('ai_model')}`\n"
        f"🧠 *AI Mode:* {mode_label}\n"
        f"🌐 *Language:* {s.get('language')}\n"
        f"🗑️ *Auto Delete:* {on_off(s.get('auto_delete'))}\n"
        f"🔔 *Notifications:* {on_off(s.get('notifications'))}\n"
        f"🔑 *Gemini Key:* {has_key}\n\n"
        "_Tap an option to change it._"
    )


DOWNLOAD_FORWARD_PROMPT = (
    "📥 *Download mode active.*\n\n"
    "Forward me a Telegram message that contains media (video, document, audio, "
    "photo, voice, sticker, GIF, video note…).\n\n"
    "I'll download it and send the file back to you.\n\n"
    f"_Max {config.max_file_size_mb} MB per file (official Bot API limit)._"
)

DOWNLOAD_CANCEL_PROMPT = (
    "⏹️ Download cancelled. Returning you to the menu."
)

ANALYZE_FORWARD_PROMPT = (
    "🎬 *AI Analyze mode active.*\n\n"
    "Forward me media matching your selected AI mode:\n"
    "• *Movie ID* → a video\n"
    "• *Transcribe* → an audio file\n"
    "• *OCR / Describe / Translate* → an image or video\n\n"
    "Pick a different mode from *🧠 Choose AI Mode* first if needed.\n\n"
    f"_Max {config.max_file_size_mb} MB (official Bot API limit)._"
)

ANALYZE_CANCEL_PROMPT = "⏹️ Analysis cancelled. Returning you to the menu."


def progress_text(label: str, percent: float, current: int, total: int,
                  speed: str = "") -> str:
    bar_len = 18
    filled = int(bar_len * max(0.0, min(1.0, percent / 100)))
    bar = "█" * filled + "░" * (bar_len - filled)
    extra = f" • {speed}" if speed else ""
    return (
        f"*{label}*\n\n"
        f"`{bar}` {percent:5.1f}%\n"
        f"{human_size(current)} / {human_size(total)}{extra}"
    )


def download_done_text(file_name: str, size: int, media_type: str) -> str:
    return (
        "✅ *Download complete!*\n\n"
        f"📄 *File:* `{file_name}`\n"
        f"📦 *Size:* {human_size(size)}\n"
        f"🗂 *Type:* {media_type}\n\n"
        "Sending the file to you now…"
    )


def download_failed_text(reason: str) -> str:
    return (
        f"❌ *Download failed.*\n\n"
        f"Reason: `{reason}`\n\n"
        "You can try again from the Download menu."
    )


def file_too_big_text(limit_mb: int) -> str:
    return (
        f"⚠️ *File too large.*\n\n"
        f"The official Telegram Bot API can download files up to "
        f"*{limit_mb} MB*. Please forward a smaller file."
    )


def analyze_result_text(r: dict) -> str:
    confidence = r.get("confidence")
    if confidence is None:
        conf_str = "—"
    else:
        conf_str = f"{confidence * 100:.0f}%"

    def cell(label: str, val) -> str:
        return f"  • *{label}:* {val if val not in (None, '', 'Unknown') else '—'}"

    return (
        "🎬 *AI Analysis Result*\n\n"
        f"{cell('Category', r.get('category'))}\n"
        f"{cell('Title', r.get('title'))}\n"
        f"{cell('Season', r.get('season'))}\n"
        f"{cell('Episode', r.get('episode'))}\n"
        f"{cell('Year', r.get('year'))}\n"
        f"{cell('Language', r.get('language'))}\n"
        f"{cell('Quality', r.get('quality'))}\n"
        f"{cell('Confidence', conf_str)}\n"
    )


def analyze_failed_text(reason: str) -> str:
    return (
        f"❌ *Analysis failed.*\n\n"
        f"Reason: `{reason}`\n\n"
        "Make sure the file is a valid, non-corrupted video within the "
        "Bot API size limit."
    )


def render_history_rows(rows: list[dict], kind: str) -> str:
    if not rows:
        return (
            f"*{kind}*\n\n"
            "No entries yet. Once you start using the bot, your recent "
            "activity will appear here."
        )
    lines = [f"*{kind} — last {len(rows)} entries*", ""]
    for r in rows:
        date = (r.get("created_at") or "")[:16].replace("T", " ")
        if kind.startswith("📥"):
            name = r.get("file_name", "?")
            status_icon = "✅" if r.get("status") == "done" else "❌"
            lines.append(
                f"{status_icon} `{date}`\n"
                f"    📄 {name}\n"
                f"    📦 {human_size(r.get('file_size', 0))} · "
                f"{r.get('media_type', '—')} · {r.get('source', '—')}"
            )
        else:
            title = r.get("title") or r.get("file_name") or "?"
            cat = r.get("category") or "—"
            conf = r.get("confidence")
            conf_s = f"{conf * 100:.0f}%" if conf is not None else "—"
            status_icon = "✅" if r.get("status") == "done" else "❌"
            lines.append(
                f"{status_icon} `{date}`\n"
                f"    🎬 {title}\n"
                f"    🏷️ {cat} · conf {conf_s}"
            )
    return "\n".join(lines)


HELP_DOWNLOAD = (
    "📥 *How to Download Media*\n\n"
    "1. Tap *📥 Download Media* in the main menu.\n"
    "2. Tap *↩️ Forward media to me*.\n"
    "3. Forward a Telegram message that contains media.\n"
    "4. Watch the live progress bar.\n"
    "5. Receive the file back — ready to save or forward.\n\n"
    "Downloads are chunked, resumable and run concurrently, so you can "
    f"queue several at once.\n\n"
    f"_Limit: {config.max_file_size_mb} MB per file (official Bot API ceiling)."
    f" Files you receive back are sent as documents (up to 50 MB)._"
)

HELP_AI = (
    "🤖 *How AI Recognition Works*\n\n"
    "1. Tap *🎬 AI Analyze* and forward a video.\n"
    "2. The bot extracts {n} evenly-spaced frames using ffmpeg.\n"
    "3. Frames are sent to Google Gemini with a structured prompt.\n"
    "4. You receive: category, title, season, episode, year, language, "
    "quality and a confidence score.\n\n"
    "Only the extracted frames (PNG/JPEG) are sent — never your full video.\n\n"
    f"_Limit: {config.max_file_size_mb} MB video (official Bot API ceiling)._"
)

HELP_FORMATS = (
    "📄 *Supported Formats*\n\n"
    "Telegram-native: photo, voice, video note, sticker, GIF/animation, video, "
    "audio, document.\n\n"
    "Common file extensions handled:\n"
    "• Videos: mp4, mkv, avi, mov, webm, flv, wmv, m4v, ts, 3gp\n"
    "• Audio: mp3, m4a, aac, flac, ogg, opus, wav\n"
    "• Archives: zip, rar, 7z, tar, gz, bz2, xz\n"
    "• Documents: pdf, epub, txt, doc, docx, mobi, azw3\n"
    "• Apps: apk, xapk, ipa, deb, rpm, exe, msi\n"
    "• Images: jpg, png, webp, bmp, gif, heic\n"
    "• Disc images: iso, img, bin\n"
    "• Subtitles: srt, ass, ssa, vtt\n\n"
    f"_All downloads use the official Bot API ({config.max_file_size_mb} MB max)._"
)

HELP_FAQ = (
    "❓ *Frequently Asked Questions*\n\n"
    "Q: Do I need to type any commands?\n"
    "A: No — everything is done through inline buttons.\n\n"
    "Q: Why can't I download a file by pasting a link?\n"
    "A: This bot uses only the official Telegram Bot API, which cannot "
    "resolve arbitrary t.me message links. Forward the media directly to the "
    "bot instead.\n\n"
    f"Q: How big a file can I download?\n"
    f"A: Up to {config.max_file_size_mb} MB — the official Bot API getFile "
    f"ceiling. Received files are sent back as documents (up to 50 MB).\n\n"
    "Q: Where are my files stored?\n"
    "A: Temporarily on the server, then auto-deleted (see Settings).\n\n"
    "Q: Which Gemini model is used?\n"
    "A: Default is gemini-2.0-flash — change it in Settings."
)


# --- Inspector (chat / user finder) ---
INSPECTOR_MENU_TEXT = (
    "*🔍 Inspect Chat / User*\n\n"
    "Look up any *public* Telegram chat, channel or user by their "
    "@username or a t.me link — the bot fetches the title, type, member "
    "count and description via the official Bot API.\n\n"
    "_Note: the Bot API can only see public chats (or private ones the bot "
    "has joined). This is a Telegram limitation, not a bot limitation._"
)

INSPECTOR_PROMPT = (
    "🔍 *Inspector mode active.*\n\n"
    "Send me one of:\n"
    "• a @username (e.g. `@durov`)\n"
    "• a public t.me link (e.g. `https://t.me/telegram`)\n\n"
    "I'll fetch everything the Bot API exposes about that chat/user."
)

INSPECTOR_EMPTY = "⚠️ Please send a @username or a public t.me link."


def inspect_result_text(info: dict) -> str:
    kind = info.get("type") or "chat"
    title = info.get("title") or info.get("first_name") or "—"
    uname = info.get("username")
    uname_s = f"@{uname}" if uname else "—"
    members = info.get("members")
    members_s = f"{members:,}" if members else "—"
    desc = (info.get("description") or "").strip()
    bio = (info.get("bio") or "").strip()
    lines = [f"🔍 *Inspection Result*", "",
             f"🏷️ *Type:* {kind}",
             f"📛 *Name:* {title}",
             f"🔗 *Username:* {uname_s}"]
    if info.get("last_name"):
        lines.append(f"👤 *Last name:* {info['last_name']}")
    if members:
        lines.append(f"👥 *Members:* {members_s}")
    if info.get("is_bot") is not None:
        lines.append(f"🤖 *Bot:* {'Yes' if info['is_bot'] else 'No'}")
    if info.get("chat_id"):
        lines.append(f"🆔 *ID:* `{info['chat_id']}`")
    if desc:
        lines.append(f"\n📝 *Description:*\n{desc[:800]}")
    if bio:
        lines.append(f"\n💬 *Bio:*\n{bio[:800]}")
    return "\n".join(lines)


def inspect_failed_text(reason: str) -> str:
    return (
        f"❌ *Inspection failed.*\n\n"
        f"Reason: `{reason}`\n\n"
        "Common causes: the username doesn't exist, the chat is private and "
        "the bot hasn't joined it, or Telegram rate-limited the request."
    )


# --- Media Toolbox ---
TOOLBOX_MENU_TEXT = (
    "*🧰 Media Toolbox*\n\n"
    "Pick a tool, then forward the media it needs. All processing runs "
    "locally with ffmpeg / Pillow — your files never leave the server "
    "except to be sent back to you."
)

TOOLBOX_PROMPTS = {
    "audio": (
        "🎵 *Extract Audio mode.*\n\n"
        "Forward a *video* — I'll extract its audio track as an MP3."
    ),
    "thumb": (
        "🖼️ *Thumbnail mode.*\n\n"
        "Forward a *video or image* — I'll grab a representative frame / "
        "thumbnail and send it back."
    ),
    "info": (
        "📋 *Media Info mode.*\n\n"
        "Forward any media — I'll show its full technical metadata "
        "(codec, resolution, bitrate, duration…)."
    ),
    "compress": (
        "🎬 *Compress Video mode.*\n\n"
        "Forward a *video* — I'll re-encode it with H.264 + a fast preset "
        "to shrink the file size."
    ),
    "imgconv": (
        "🔄 *Convert Image mode.*\n\n"
        "Forward an *image* — reply with the target format "
        "(`png`, `jpg`, `webp`, `bmp`) and I'll convert it."
    ),
}

TOOLBOX_INVALID_MEDIA = "⚠️ That tool needs a different kind of media. Please forward the right type."


def media_info_text(info: dict) -> str:
    lines = ["📋 *Media Information*", ""]
    for k, v in info.items():
        if v in (None, "", []):
            continue
        lines.append(f"• *{k}:* `{v}`")
    return "\n".join(lines)


# --- Library ---
LIBRARY_MENU_TEXT = (
    "*⭐ Library*\n\n"
    "Your bookmarked media. After any download, tap *Save to Library* to "
    "keep a record (filename, type, size, note) you can browse & search later."
)

LIBRARY_EMPTY = (
    "⭐ *Your library is empty.*\n\n"
    "After downloading media, tap *Save to Library* to bookmark it here."
)


def library_entry_text(entry: dict) -> str:
    date = (entry.get("created_at") or "")[:16].replace("T", " ")
    note = entry.get("note") or ""
    tags = entry.get("tags") or ""
    lines = [
        "⭐ *Library Entry*", "",
        f"📄 *File:* `{entry.get('file_name','?')}`",
        f"🗂 *Type:* {entry.get('media_type','—')}",
        f"📦 *Size:* {human_size(entry.get('file_size',0))}",
        f"🕒 *Saved:* `{date}`",
    ]
    if tags:
        lines.append(f"🏷️ *Tags:* {tags}")
    if note:
        lines.append(f"\n📝 *Note:*\n{note}")
    return "\n".join(lines)


def library_list_text(rows: list[dict], title: str) -> str:
    if not rows:
        return f"*{title}*\n\n{LIBRARY_EMPTY.split(chr(10),1)[1]}"
    lines = [f"*{title} — {len(rows)} entries*", ""]
    for i, r in enumerate(rows, 1):
        date = (r.get("created_at") or "")[:10]
        lines.append(
            f"{i}. `{r.get('file_name','?')}`\n"
            f"    📦 {human_size(r.get('file_size',0))} · "
            f"{r.get('media_type','—')} · {date}"
        )
    return "\n".join(lines)


# --- Stats ---
STATS_MENU_TEXT = "*📊 Statistics*\n\nView your activity or global bot stats."


def user_stats_text(s: dict) -> str:
    by_type = s.get("by_type") or []
    type_lines = "\n".join(f"  • {t or '—'}: {n}" for t, n in by_type) or "  • —"
    return (
        "👤 *Your Stats*\n\n"
        f"📥 *Downloads:* {s.get('downloads',0)}\n"
        f"📦 *Downloaded:* {human_size(s.get('download_bytes',0))}\n"
        f"🤖 *AI Analyses:* {s.get('analyses',0)}\n"
        f"⭐ *Library entries:* {s.get('library',0)}\n"
        f"🔍 *Inspected chats:* {s.get('inspected',0)}\n\n"
        f"📊 *Top media types:*\n{type_lines}"
    )


def global_stats_text(s: dict) -> str:
    return (
        "🌍 *Global Stats*\n\n"
        f"👥 *Users:* {s.get('users',0)}\n"
        f"📥 *Downloads:* {s.get('downloads',0)}\n"
        f"📦 *Downloaded:* {human_size(s.get('download_bytes',0))}\n"
        f"🤖 *AI Analyses:* {s.get('analyses',0)}\n"
        f"⭐ *Library entries:* {s.get('library',0)}\n"
        f"🔍 *Inspected chats:* {s.get('inspected',0)}\n"
    )


# --- QR Code ---
QR_MENU_TEXT = (
    "*🔳 QR Code Generator*\n\n"
    "Turn any text or link into a scannable QR code — perfect for sharing "
    "URLs, Wi-Fi credentials, contacts or commands."
)

QR_PROMPT = (
    "🔳 *QR mode active.*\n\n"
    "Send me the text or link you want encoded into a QR code."
)


# --- Quota ---
def quota_exceeded_text(count_limit: int, bytes_limit_mb: int) -> str:
    return (
        f"⚠️ *Daily quota reached.*\n\n"
        f"You've hit your daily limit of *{count_limit} downloads* or "
        f"*{bytes_limit_mb} MB*. Please come back tomorrow."
    )


# --- New help section ---
HELP_FEATURES = (
    "🔍 *Inspect Chat / User*\n\n"
    "Tap *🔍 Inspect Chat*, then send a @username or public t.me link. The "
    "bot fetches the chat's title, type, member count and description via "
    "the official Bot API. Only *public* chats are visible (a Telegram "
    "limitation).\n\n"
    "🧰 *Media Toolbox*\n\n"
    "• *Extract Audio (MP3)* — pull the audio track out of a video.\n"
    "• *Extract Thumbnail* — grab a representative frame.\n"
    "• *Media Info* — full technical metadata (codec, resolution, bitrate…).\n"
    "• *Compress Video* — re-encode to shrink file size.\n"
    "• *Convert Image* — PNG ⇄ JPG ⇄ WEBP ⇄ BMP.\n\n"
    "⭐ *Library*\n\n"
    "Bookmark any download to your personal, searchable library.\n\n"
    "📊 *Stats* — your activity and global bot counters.\n\n"
    "🔳 *QR Code* — encode any text/link into a QR code."
)


# --- AI modes ---
AI_MODES_TEXT = (
    "*🧠 AI Mode*\n\n"
    "Pick what you want the AI to do. Each mode accepts different media:\n\n"
    "🎬 *Movie / Series ID* — identify a film/series/anime from video frames\n"
    "🎵 *Audio Transcription* — transcribe speech from an audio file\n"
    "🔤 *Text Extraction (OCR)* — pull all visible text from images/frames\n"
    "📝 *Scene Description* — describe what's in images/frames\n"
    "🌐 *Image Translation* — OCR + translate text to your language\n\n"
    "The selected mode is used when you tap *Forward media to analyze*."
)


def ai_mode_set_text(label: str) -> str:
    return f"✅ *AI mode set to:* {label}\n\nNow forward the matching media type."


# --- Batch / album ---
BATCH_MENU_TEXT = (
    "*📦 Batch / Album*\n\n"
    "Forward me a *media group* (Telegram album — up to 10 photos/videos sent "
    "together) or several media messages in a row.\n\n"
    "I'll download them all and send them back as a single album."
)

BATCH_PROMPT = (
    "📦 *Batch mode active.*\n\n"
    "Forward a media group (album) or several media messages now. "
    "Send them one after another within ~5 seconds."
)

BATCH_TIMEOUT = "⏱️ Batch window closed. Processing what was received…"
BATCH_EMPTY = "⚠️ No media received during the batch window. Try again."


def batch_summary_text(count: int, total_size: int) -> str:
    return (
        f"📦 *Batch complete!*\n\n"
        f"Files processed: *{count}*\n"
        f"Total size: *{human_size(total_size)}*\n\n"
        "Sending them as an album now…"
    )


def batch_failed_text(reason: str) -> str:
    return f"❌ *Batch failed.*\n\nReason: `{reason}`"


# --- Scheduled downloads ---
SCHEDULED_MENU_TEXT = (
    "*⏰ Scheduled Tasks*\n\n"
    "Queue a download or analysis to run later.\n\n"
    "How to schedule: open *📥 Download*, forward media, then tap "
    "*Schedule instead* to pick a time."
)


def scheduled_list_text(rows: list[dict]) -> str:
    if not rows:
        return "⏰ *Scheduled Tasks*\n\nNo scheduled tasks. Queue one from Download."
    lines = ["⏰ *Scheduled Tasks*", ""]
    for r in rows:
        run_at = (r.get("run_at") or "")[:16].replace("T", " ")
        kind = r.get("kind", "?")
        status = r.get("status", "?")
        icon = {"pending": "⏳", "done": "✅", "failed": "❌"}.get(status, "•")
        lines.append(f"{icon} #{r['id']} `{run_at}` — {kind} ({status})")
    return "\n".join(lines)


SCHEDULE_PROMPT = (
    "⏰ *Schedule mode.*\n\n"
    "Send a time in the format `YYYY-MM-DD HH:MM` (e.g. `2025-12-31 23:00`), "
    "then forward the media to queue."
)

SCHEDULE_INVALID = "⚠️ Invalid time. Use `YYYY-MM-DD HH:MM` (24h, e.g. 2025-12-31 23:00)."


# --- Backup / restore ---
BACKUP_MENU_TEXT = (
    "*💾 Backup & Restore*\n\n"
    "Export your settings, history, library and inspections as a JSON file. "
    "Import a previous backup to restore settings."
)

BACKUP_EXPORTING = "💾 Exporting your data…"
BACKUP_IMPORT_PROMPT = (
    "📥 Send me the backup JSON file to import your settings."
)


def backup_export_done_text(file_name: str, size: int, counts: dict) -> str:
    parts = []
    for k, v in counts.items():
        parts.append(f"  • {k}: {v}")
    return (
        f"✅ *Backup exported!*\n\n"
        f"📄 `{file_name}` ({human_size(size)})\n\n"
        f"Contents:\n" + "\n".join(parts)
    )


def backup_import_done_text(summary: dict) -> str:
    return (
        f"✅ *Import complete!*\n\n"
        f"Settings restored: {summary.get('settings', 0)} fields"
    )


def backup_failed_text(reason: str) -> str:
    return f"❌ *Backup failed:* `{reason}`"


# --- Admin ---
ADMIN_DENIED = (
    "🚫 *Admin only.*\n\nYou do not have permission to use this section."
)

ADMIN_MENU_TEXT = (
    "*🛡️ Admin Panel*\n\n"
    "Bot management tools (admin only)."
)

ADMIN_BCAST_PROMPT = (
    "📣 *Broadcast mode.*\n\n"
    "Send the message text you want to broadcast to all bot users."
)


def admin_users_text(users: list[dict], total: int) -> str:
    if not users:
        return "👥 *Users*\n\nNo users yet."
    lines = [f"👥 *Users ({total} total)*", ""]
    for u in users[:20]:
        name = u.get("first_name") or u.get("username") or str(u.get("tg_id"))
        uname = f"@{u['username']}" if u.get("username") else "—"
        admin = " 🛡️" if u.get("is_admin") else ""
        lines.append(f"• {name}{admin}\n   {uname} · `{u['tg_id']}`")
    if total > 20:
        lines.append(f"\n_…and {total - 20} more_")
    return "\n".join(lines)


def admin_bcast_done_text(sent: int, failed: int) -> str:
    return (
        f"📣 *Broadcast complete!*\n\n"
        f"✅ Sent: {sent}\n"
        f"❌ Failed: {failed}"
    )


# --- Inline mode ---
def inline_help_text() -> str:
    return (
        "🎬 *MediaGrab AI Bot*\n\n"
        "Available inline queries:\n"
        "• `@bot help` — this help\n"
        "• `@bot qr <text>` — generate a QR code\n"
        "• `@bot info` — your quick stats\n\n"
        "Add me to a chat and tap /start to use all features."
    )


def inline_stats_text(s: dict) -> str:
    return (
        f"📊 Your stats:\n"
        f"📥 Downloads: {s.get('downloads',0)}\n"
        f"🤖 Analyses: {s.get('analyses',0)}\n"
        f"⭐ Library: {s.get('library',0)}"
    )


# ===========================================================================
# Link / username download texts (Task ID 5)
# ===========================================================================

LINK_DOWNLOAD_PROMPT = (
    "🔗 *Download by link or @username*\n\n"
    "Send me any of:\n"
    "• A message link: `https://t.me/channel/123`\n"
    "• A private link: `https://t.me/c/1234567890/123`\n"
    "• A @username: `@channelusername`\n"
    "• A channel link: `https://t.me/channel`\n\n"
    "_For public channels I can fetch the message directly. For private "
    "links (t.me/c/…) the bot must be a member of that chat._"
)

LINK_DOWNLOAD_EMPTY = (
    "⚠️ Please send a Telegram message link or @username."
)


def link_parse_error_text(reason: str) -> str:
    return f"❌ *Could not parse that input.*\n\n`{reason}`"


def link_resolving_text(description: str) -> str:
    return f"📥 Resolving: {description}…"


def link_download_failed_text(reason: str) -> str:
    return (
        f"❌ *Download failed.*\n\n"
        f"Reason: `{reason}`\n\n"
        "Common causes:\n"
        "• The message was deleted or doesn't exist\n"
        "• The chat is private and the bot isn't a member\n"
        "• The message has no downloadable media"
    )


def link_resolved_ask_msgid(info: dict) -> str:
    """Show resolved chat info and ask for a message id."""
    title = info.get("title") or info.get("first_name") or "—"
    uname = info.get("username")
    uname_s = f"@{uname}" if uname else "—"
    kind = info.get("type") or "chat"
    members = info.get("members")
    members_s = f"{members:,}" if members else "—"
    desc = (info.get("description") or "").strip()

    lines = [
        "✅ *Chat resolved!*\n",
        f"📛 *Name:* {title}",
        f"🏷️ *Type:* {kind}",
        f"🔗 *Username:* {uname_s}",
    ]
    if members:
        lines.append(f"👥 *Members:* {members_s}")
    if desc:
        lines.append(f"\n📝 *Description:*\n{desc[:400]}")
    lines.append("")
    lines.append(
        "📝 *Now send me the message id* (a number, e.g. `123`).\n"
        "You can find it as the last part of a t.me link, or in the "
        "channel's message URL."
    )
    return "\n".join(lines)


# ===========================================================================
# States
# ===========================================================================

IDLE = "idle"
AWAIT_DOWNLOAD_FORWARD = "await_download_forward"
AWAIT_ANALYZE = "await_analyze"
AWAIT_INSPECT = "await_inspect"
AWAIT_TOOLBOX = "await_toolbox"          # value stored as "tb:<tool>"
AWAIT_LIBRARY_SEARCH = "await_library_search"
AWAIT_QR = "await_qr"
AWAIT_BATCH = "await_batch"              # collecting a media group
AWAIT_SCHEDULE = "await_schedule"        # awaiting a time + media
AWAIT_BACKUP_IMPORT = "await_backup_import"
AWAIT_ADMIN_BCAST = "await_admin_bcast"
AWAIT_LINK_DOWNLOAD = "await_link_download"      # link/username input
AWAIT_LINK_MESSAGE_ID = "await_link_message_id"  # message id after resolve
AWAIT_MTPROTO_SCREENSHOT = "await_mtproto_screenshot"
AWAIT_MTPROTO_DOWNLOAD = "await_mtproto_download"
AWAIT_MTPROTO_MSGID = "await_mtproto_msgid"   # picking a message id after history
AWAIT_MEDIA_BROWSE = "await_media_browse"     # user entered a channel to browse
AWAIT_VC_JOIN_TARGET = "await_vc_join_target"
AWAIT_VC_CHECK_TARGET = "await_vc_check_target"
AWAIT_VC_STAY = "await_vc_stay"

_states: dict[int, str] = defaultdict(lambda: IDLE)
_tool: dict[int, str] = {}               # user_id -> active toolbox tool
_last_download: dict[int, dict] = {}     # user_id -> last completed download meta

_states_lock = asyncio.Lock()


async def set_state(user_id: int, state: str) -> None:
    async with _states_lock:
        _states[user_id] = state


async def get_state(user_id: int) -> str:
    return _states.get(user_id, IDLE)


async def reset(user_id: int) -> None:
    async with _states_lock:
        _states[user_id] = IDLE
        _tool.pop(user_id, None)


async def set_tool(user_id: int, tool: str) -> None:
    async with _states_lock:
        _tool[user_id] = tool


async def get_tool(user_id: int) -> str:
    return _tool.get(user_id, "")


async def set_last_download(user_id: int, meta: dict) -> None:
    async with _states_lock:
        _last_download[user_id] = meta


async def get_last_download(user_id: int) -> dict | None:
    return _last_download.get(user_id)


def set_state_sync(user_id: int, state: str) -> None:
    _states[user_id] = state


def get_state_sync(user_id: int) -> str:
    return _states.get(user_id, IDLE)


def set_tool_sync(user_id: int, tool: str) -> None:
    _tool[user_id] = tool


def get_tool_sync(user_id: int) -> str:
    return _tool.get(user_id, "")


def set_last_download_sync(user_id: int, meta: dict) -> None:
    _last_download[user_id] = meta


def get_last_download_sync(user_id: int) -> dict | None:
    return _last_download.get(user_id)


# ===========================================================================
# Services — MTProto backend (Telethon, optional)
# ===========================================================================
# All Telethon imports are done lazily inside functions so the bot boots even
# when Telethon isn't installed. Type hints work because of
# ``from __future__ import annotations``.
#
# This section consolidates:
#   * services/session_manager.py     — session file lifecycle
#   * services/mtproto_manager.py     — singleton TelegramClient + watchdog
#   * services/mtproto_service.py     — clean call interface (resolve / dl / shot)
#   * services/mtproto_capture.py     — self-destruct media capture handler
# ===========================================================================


# --- session_manager -------------------------------------------------------

_MTPROTO_SESSION_DIR = config.project_root / "data"


def session_path() -> Path:
    """Return the full path to the Telethon session file (without extension)."""
    return _MTPROTO_SESSION_DIR / config.tg_user_session


def session_file_exists() -> bool:
    """True if the ``.session`` file exists and is non-empty."""
    p = str(session_path()) + ".session"
    return os.path.exists(p) and os.path.getsize(p) > 0


async def validate_session() -> bool:
    """Check whether the stored session can authenticate."""
    if not config.mtproto_configured:
        return False
    if not session_file_exists():
        return False
    try:
        from telethon import TelegramClient
        client = TelegramClient(str(session_path()), config.tg_api_id,
                                config.tg_api_hash)
        await client.connect()
        if not await client.is_user_authorized():
            await client.disconnect()
            return False
        me = await client.get_me()
        await client.disconnect()
        return me is not None
    except Exception as exc:  # noqa: BLE001
        logger.warning("session validation failed: %s", exc)
        return False


async def interactive_login(phone: str | None = None) -> str:
    """Run an interactive login flow to create the session file.

    Call this from a terminal (not from the bot) the first time:
        python -m services.session_manager
    """
    if not config.tg_api_id or not config.tg_api_hash:
        return "ERROR: TG_API_ID and TG_API_HASH must be set in .env"

    from telethon import TelegramClient
    _MTPROTO_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    client = TelegramClient(str(session_path()), config.tg_api_id,
                            config.tg_api_hash)
    await client.start(phone=phone)
    me = await client.get_me()
    status = (f"✅ Login successful!\n\n"
              f"Logged in as: @{getattr(me, 'username', '—')} "
              f"({getattr(me, 'first_name', '')} {getattr(me, 'last_name', '')})\n"
              f"User ID: {me.id}\n"
              f"Session saved to: {session_path()}.session")
    await client.disconnect()
    logger.info("MTProto session created for user %s", me.id)
    return status


def session_info() -> dict:
    """Return static info about the session configuration."""
    return {
        "enabled": config.mtproto_enabled,
        "configured": config.mtproto_configured,
        "session_name": config.tg_user_session,
        "session_path": str(session_path()) + ".session",
        "session_exists": session_file_exists(),
        "api_id_set": bool(config.tg_api_id),
        "api_hash_set": bool(config.tg_api_hash),
    }


# --- mtproto_manager -------------------------------------------------------

_mtproto_client: Optional[Any] = None  # TelegramClient
_mtproto_started: bool = False
_mtproto_start_time: float = 0.0
_mtproto_health_task: Optional[asyncio.Task] = None
_mtproto_watchdog_task: Optional[asyncio.Task] = None
_mtproto_self_destruct_registered: bool = False

_mtproto_stats = {
    "requests": 0,
    "successes": 0,
    "failures": 0,
    "flood_waits": 0,
    "reconnects": 0,
}


def mtproto_is_available() -> bool:
    """True if Telethon is importable."""
    try:
        import telethon  # noqa: F401
        return True
    except ImportError:
        return False


def mtproto_is_started() -> bool:
    return (_mtproto_started
            and _mtproto_client is not None
            and _mtproto_client.is_connected())


async def mtproto_start() -> str:
    """Start the MTProto client. Returns a status string."""
    global _mtproto_client, _mtproto_started, _mtproto_start_time

    if _mtproto_started and _mtproto_client is not None and _mtproto_client.is_connected():
        return "already_running"
    if not config.mtproto_configured:
        return "not_configured"
    if not mtproto_is_available():
        return "telethon_not_installed"

    from telethon import TelegramClient

    if not session_file_exists():
        logger.warning("MTProto session file not found — run interactive login first.")
        return "no_session_file"

    try:
        _mtproto_client = TelegramClient(
            str(session_path()),
            config.tg_api_id,
            config.tg_api_hash,
            connection_retries=5,
            retry_delay=2,
            auto_reconnect=True,
            request_retries=3,
        )
        await _mtproto_client.connect()
        if not await _mtproto_client.is_user_authorized():
            logger.error("MTProto session not authorized — re-login required.")
            await _mtproto_client.disconnect()
            _mtproto_client = None
            return "not_authorized"

        me = await _mtproto_client.get_me()
        _mtproto_started = True
        _mtproto_start_time = time.time()
        logger.info("MTProto client started as @%s (id=%s)",
                    getattr(me, "username", None), me.id)

        await _mtproto_register_handlers()
        _mtproto_start_background_tasks()

        return f"started:{getattr(me, 'username', me.id)}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("MTProto start failed: %s", exc)
        _mtproto_client = None
        _mtproto_started = False
        return f"error:{exc}"


async def mtproto_stop() -> str:
    """Disconnect the MTProto client and cancel background tasks."""
    global _mtproto_client, _mtproto_started, _mtproto_health_task, _mtproto_watchdog_task

    if _mtproto_health_task and not _mtproto_health_task.done():
        _mtproto_health_task.cancel()
        _mtproto_health_task = None
    if _mtproto_watchdog_task and not _mtproto_watchdog_task.done():
        _mtproto_watchdog_task.cancel()
        _mtproto_watchdog_task = None

    if _mtproto_client is not None:
        try:
            await _mtproto_client.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.debug("MTProto disconnect error: %s", exc)
        _mtproto_client = None
    _mtproto_started = False
    logger.info("MTProto client stopped.")
    return "stopped"


async def mtproto_restart() -> str:
    """Restart the MTProto client."""
    await mtproto_stop()
    await asyncio.sleep(1)
    return await mtproto_start()


def mtproto_get_client() -> Optional[Any]:
    """Return the active TelegramClient, or None if not started."""
    return _mtproto_client if mtproto_is_started() else None


def mtproto_get_status() -> dict[str, Any]:
    """Return a status snapshot for the admin panel."""
    return {
        "available": mtproto_is_available(),
        "configured": config.mtproto_configured,
        "started": mtproto_is_started(),
        "uptime_seconds": (time.time() - _mtproto_start_time) if _mtproto_started else 0,
        "stats": dict(_mtproto_stats),
        "session": session_info(),
    }


async def mtproto_call_with_retry(func: Callable, *args, retries: int = 3,
                                  **kwargs) -> Any:
    """Call a Telethon coroutine with FloodWait + network retry."""
    from telethon.errors import FloodWaitError

    _mtproto_stats["requests"] += 1
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            result = await func(*args, **kwargs)
            _mtproto_stats["successes"] += 1
            return result
        except FloodWaitError as exc:
            _mtproto_stats["flood_waits"] += 1
            wait = min(exc.seconds + 1, 60)
            logger.warning("FloodWait %ss on MTProto call (attempt %d/%d)",
                           exc.seconds, attempt, retries)
            await asyncio.sleep(wait)
            last_exc = exc
            continue
        except asyncio.CancelledError:
            raise
        except ConnectionError as exc:
            logger.warning("MTProto connection error (attempt %d/%d): %s",
                           attempt, retries, exc)
            if _mtproto_client and not _mtproto_client.is_connected():
                try:
                    await _mtproto_client.connect()
                    _mtproto_stats["reconnects"] += 1
                except Exception:  # noqa: BLE001
                    pass
            last_exc = exc
            await asyncio.sleep(2)
            continue
        except Exception as exc:  # noqa: BLE001
            _mtproto_stats["failures"] += 1
            raise
    _mtproto_stats["failures"] += 1
    raise last_exc  # type: ignore[misc]


def _mtproto_start_background_tasks() -> None:
    global _mtproto_health_task, _mtproto_watchdog_task
    _mtproto_health_task = asyncio.create_task(_mtproto_health_loop())
    _mtproto_watchdog_task = asyncio.create_task(_mtproto_watchdog_loop())


async def _mtproto_health_loop() -> None:
    """Periodically ping the MTProto client to confirm it's alive."""
    logger.info("MTProto health monitor started (interval=120s).")
    while True:
        try:
            await asyncio.sleep(120)
            if _mtproto_client and _mtproto_client.is_connected():
                await _mtproto_client.get_me()
                logger.debug("MTProto health check: OK")
            else:
                logger.warning("MTProto health check: client disconnected")
        except asyncio.CancelledError:
            logger.info("MTProto health monitor cancelled.")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("MTProto health check failed: %s", exc)


async def _mtproto_watchdog_loop() -> None:
    """Watchdog — if the client disconnects, attempt reconnection."""
    logger.info("MTProto watchdog started (interval=60s).")
    while True:
        try:
            await asyncio.sleep(60)
            if _mtproto_started and _mtproto_client and not _mtproto_client.is_connected():
                logger.warning("MTProto client disconnected — attempting reconnect.")
                try:
                    await _mtproto_client.connect()
                    if _mtproto_client.is_connected():
                        _mtproto_stats["reconnects"] += 1
                        logger.info("MTProto reconnected successfully.")
                    else:
                        logger.error("MTProto reconnect failed.")
                except Exception as exc:  # noqa: BLE001
                    logger.error("MTProto reconnect error: %s", exc)
        except asyncio.CancelledError:
            logger.info("MTProto watchdog cancelled.")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("MTProto watchdog error: %s", exc)


async def _mtproto_register_handlers() -> None:
    """Register Telethon event handlers (self-destruct media capture)."""
    global _mtproto_self_destruct_registered
    if _mtproto_self_destruct_registered or _mtproto_client is None:
        return

    from telethon import events

    @_mtproto_client.on(events.NewMessage(incoming=True))
    async def _on_new_message(event):  # type: ignore[no-untyped-def]
        try:
            await mtproto_handle_incoming_message(event, _mtproto_client)
        except Exception as exc:  # noqa: BLE001
            logger.debug("mtproto capture handler error: %s", exc)

    _mtproto_self_destruct_registered = True
    logger.info("MTProto self-destruct capture handler registered.")


# --- mtproto_capture -------------------------------------------------------


def _mtproto_is_self_destruct(message) -> bool:
    """True if the message carries self-destructing (timer) media."""
    media = getattr(message, "media", None)
    if media is None:
        return False
    photo = getattr(media, "photo", None)
    if photo and getattr(photo, "ttl_seconds", None):
        return True
    doc = getattr(media, "document", None)
    if doc:
        if getattr(media, "ttl_seconds", None):
            return True
        from telethon.tl.types import DocumentAttributeVideo
        for attr in getattr(doc, "attributes", []) or []:
            if isinstance(attr, DocumentAttributeVideo) and getattr(media, "ttl_seconds", None):
                return True
    return getattr(media, "ttl_seconds", None) is not None


def _mtproto_media_kind(message) -> str:
    """Return 'photo' | 'video' | 'document' for the message media."""
    media = getattr(message, "media", None)
    if media is None:
        return "none"
    if getattr(media, "photo", None):
        return "photo"
    doc = getattr(media, "document", None)
    if doc:
        from telethon.tl.types import DocumentAttributeVideo, DocumentAttributeAnimated
        for attr in getattr(doc, "attributes", []) or []:
            if isinstance(attr, DocumentAttributeVideo):
                if getattr(attr, "round_message", False):
                    return "video_note"
                return "video"
            if isinstance(attr, DocumentAttributeAnimated):
                return "gif"
        return "document"
    return "unknown"


async def mtproto_handle_incoming_message(event, client) -> None:
    """Telethon NewMessage handler — captures self-destruct media."""
    message = event.message
    if not _mtproto_is_self_destruct(message):
        return

    sender = await event.get_sender()
    sender_name = (getattr(sender, "username", None)
                   or f"{getattr(sender, 'first_name', '')} {getattr(sender, 'last_name', '')}".strip()
                   or str(getattr(sender, "id", "unknown")))
    sender_id = getattr(sender, "id", "unknown")
    kind = _mtproto_media_kind(message)
    ttl = getattr(message.media, "ttl_seconds", "?")

    # Try to get chat info for the admin notification.
    chat_info = await _mtproto_get_chat_info(event, client)

    logger.info("Self-destruct %s captured from @%s (id=%s, ttl=%ss)",
                kind, sender_name, sender_id, ttl)

    out_dir = config.downloads_dir / "self_destruct"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    base_name = f"sd_{kind}_{sender_id}_{timestamp}"
    target = unique_path(out_dir, base_name)

    try:
        path = await client.download_media(message, file=str(target))
        if not path:
            logger.warning("Self-destruct download returned no path.")
            return
        path = Path(path)
        size = path.stat().st_size if path.exists() else 0
        logger.info("Self-destruct media saved: %s (%d bytes)", path.name, size)

        await _mtproto_notify_admin(
            path, sender_name, sender_id, kind, ttl, message, chat_info
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Self-destruct capture failed: %s", exc)


async def _mtproto_get_chat_info(event, client) -> dict:
    """Extract chat metadata for the admin notification."""
    info: dict = {}
    try:
        chat = await event.get_chat()
        info["chat_id"] = getattr(chat, "id", None)
        info["chat_title"] = (getattr(chat, "title", None)
                              or getattr(chat, "first_name", None)
                              or getattr(chat, "username", None)
                              or str(info.get("chat_id", "unknown")))
        info["chat_type"] = type(chat).__name__
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not get chat info: %s", exc)
    return info


async def _mtproto_notify_admin(path: Path, sender_name: str, sender_id: int,
                                kind: str, ttl: Any, message,
                                chat_info: dict | None = None) -> None:
    """Send the captured media to the admin via the Bot API.

    Uses the bot registry (``get_bot``) to obtain the running Bot instance —
    NOT ``Application.get_application()`` which doesn't exist. Includes
    sender info, chat info, timestamps, and filename.
    """
    admin_id = config.mtproto_admin_id
    if not admin_id:
        logger.debug("No MTPROTO_ADMIN_ID set — skipping admin notification.")
        return

    bot = get_bot()
    if bot is None:
        logger.warning(
            "Bot instance not registered yet — cannot notify admin. "
            "Ensure register_bot(app.bot) is called in _post_init."
        )
        return

    # Build a rich caption with all available metadata.
    capture_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    msg_date = getattr(message, "date", None)
    msg_ts = msg_date.strftime("%Y-%m-%d %H:%M:%S UTC") if msg_date else "—"

    chat_info = chat_info or {}
    chat_title = md_escape(chat_info.get("chat_title", "—"))
    chat_id = chat_info.get("chat_id", "—")
    chat_type = chat_info.get("chat_type", "—")

    caption = (
        f"📸 *Self-destruct media captured*\n\n"
        f"👤 *From:* @{md_escape(sender_name)} (`{sender_id}`)\n"
        f"💬 *Chat:* {chat_title} (`{chat_id}`)\n"
        f"🗂 *Chat type:* {chat_type}\n"
        f"🎞 *Media type:* {kind}\n"
        f"⏱ *TTL:* {ttl}s\n"
        f"📅 *Message time:* {msg_ts}\n"
        f"🕒 *Captured at:* {capture_ts}\n"
        f"📦 *File:* `{md_escape(path.name)}`"
    )

    try:
        size = path.stat().st_size if path.exists() else 0
        if size <= config.upload_limit_bytes:
            with path.open("rb") as fh:
                await safe_send_document(
                    bot, admin_id, fh,
                    filename=path.name,
                    caption=caption,
                )
            logger.info("Self-destruct media sent to admin %s (%d bytes)",
                        admin_id, size)
        else:
            await safe_send_message(
                bot, admin_id,
                caption + "\n\n_ℹ️ File too large to send via Bot API — "
                "saved on server._",
            )
            logger.info("Self-destruct media too large for Bot API — "
                        "admin notified, file saved on server.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Admin notification failed: %s", exc)


async def mtproto_take_screenshot(chat_ref: str | int) -> Path | None:
    """Capture a screenshot of a chat's recent messages (MTProto)."""
    try:
        client = mtproto_get_client()
        if client is None:
            logger.warning("MTProto not started — cannot take screenshot.")
            return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot get MTProto client for screenshot: %s", exc)
        return None

    try:
        entity = await client.get_entity(chat_ref)
        messages = await client.get_messages(entity, limit=15)
        lines = [f"📱 Screenshot of {getattr(entity, 'title', str(chat_ref))}", ""]
        for m in reversed(messages):
            sender = await m.get_sender()
            name = (getattr(sender, "first_name", None)
                    or getattr(sender, "title", None)
                    or getattr(sender, "username", None)
                    or str(getattr(sender, "id", "?")))
            text = m.text or "(media)"
            ts = m.date.strftime("%H:%M") if m.date else "?"
            lines.append(f"[{ts}] {name}: {text[:120]}")
        content = "\n".join(lines)

        from PIL import Image, ImageDraw, ImageFont
        import textwrap

        img = Image.new("RGB", (800, 600), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
            )
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()

        y = 10
        for line in content.split("\n"):
            for wrapped in textwrap.wrap(line, width=70) or [""]:
                draw.text((10, y), wrapped, fill=(220, 220, 220), font=font)
                y += 18
                if y > 580:
                    break
            if y > 580:
                break

        out_dir = config.frames_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out = unique_path(out_dir, f"screenshot_{int(time.time())}.png")
        img.save(str(out), "PNG")
        logger.info("Screenshot saved: %s", out.name)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.exception("Screenshot failed: %s", exc)
        return None


# --- mtproto_service -------------------------------------------------------


class MTProtoError(Exception):
    """User-facing error from the MTProto backend."""


class MTProtoNotAvailable(MTProtoError):
    """Raised when MTProto is disabled or not started."""


def _mtproto_require_client():
    """Return the active Telethon client or raise."""
    client = mtproto_get_client()
    if client is None:
        raise MTProtoNotAvailable(
            "MTProto backend is not running. Enable MTPROTO_ENABLED in .env "
            "and create a session first."
        )
    return client


def mtproto_service_is_available() -> bool:
    """True if MTProto is enabled, installed, and connected."""
    return mtproto_is_started()


async def mtproto_resolve_entity(chat_ref: str | int) -> dict[str, Any]:
    """Resolve a @username or chat id to a plain dict via MTProto."""
    client = _mtproto_require_client()
    from telethon.errors import UsernameInvalidError, ChannelInvalidError

    try:
        entity = await mtproto_call_with_retry(client.get_entity, chat_ref)
    except UsernameInvalidError as exc:
        raise MTProtoError(f"Username '{chat_ref}' is invalid or doesn't exist.") from exc
    except ChannelInvalidError as exc:
        raise MTProtoError(f"Channel '{chat_ref}' is not accessible.") from exc
    except Exception as exc:  # noqa: BLE001
        raise MTProtoError(f"Failed to resolve '{chat_ref}': {exc}") from exc

    return {
        "id": getattr(entity, "id", None),
        "username": getattr(entity, "username", None),
        "title": getattr(entity, "title", None),
        "first_name": getattr(entity, "first_name", None),
        "last_name": getattr(entity, "last_name", None),
        "type": _mtproto_entity_type(entity),
        "participants_count": getattr(entity, "participants_count", None),
    }


def _mtproto_entity_type(entity) -> str:
    from telethon.tl.types import User, Chat, Channel
    if isinstance(entity, User):
        return "user"
    if isinstance(entity, Chat):
        return "group"
    if isinstance(entity, Channel):
        return "channel" if getattr(entity, "broadcast", False) else "supergroup"
    return "unknown"


async def mtproto_get_channel_history(chat_ref: str | int, limit: int = 20,
                                      offset_id: int = 0) -> list[dict[str, Any]]:
    """Fetch recent messages from a channel/chat via MTProto."""
    client = _mtproto_require_client()

    try:
        entity = await mtproto_call_with_retry(client.get_entity, chat_ref)
        messages = await mtproto_call_with_retry(
            client.get_messages, entity, limit=limit, offset_id=offset_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise MTProtoError(f"Failed to read history: {exc}") from exc

    out = []
    for m in messages:
        out.append({
            "id": m.id,
            "date": m.date.isoformat() if m.date else None,
            "text": (m.text or "")[:500],
            "has_media": m.media is not None,
            "media_type": _mtproto_media_type(m.media),
            "sender_id": getattr(m, "sender_id", None),
        })
    return out


def _mtproto_media_type(media) -> Optional[str]:
    if media is None:
        return None
    if getattr(media, "photo", None):
        return "photo"
    doc = getattr(media, "document", None)
    if doc:
        from telethon.tl.types import DocumentAttributeVideo, DocumentAttributeAudio
        for attr in getattr(doc, "attributes", []) or []:
            if isinstance(attr, DocumentAttributeVideo):
                return "video_note" if getattr(attr, "round_message", False) else "video"
            if isinstance(attr, DocumentAttributeAudio):
                return "voice" if getattr(attr, "voice", False) else "audio"
        return "document"
    return "unknown"


async def mtproto_download_message_media(chat_ref: str | int, message_id: int,
                                         progress_cb=None) -> tuple[Path, str, str, int]:
    """Download media from a specific message via MTProto.

    Returns ``(path, filename, media_type, size_bytes)``.
    """
    client = _mtproto_require_client()

    try:
        entity = await mtproto_call_with_retry(client.get_entity, chat_ref)
        messages = await mtproto_call_with_retry(
            client.get_messages, entity, ids=message_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise MTProtoError(f"Failed to fetch message #{message_id}: {exc}") from exc

    # Telethon returns a single Message object (not a list) when ids=<int>.
    # Normalise to a single message, handling list, single Message, and None.
    message = _mtproto_extract_single_message(messages)
    if message is None:
        raise MTProtoError(f"Message #{message_id} not found in that chat.")
    if not message.media:
        raise MTProtoError(f"Message #{message_id} has no downloadable media.")

    media_type = _mtproto_media_type(message.media) or "media"
    filename = _mtproto_derive_filename(message, media_type)
    filename = safe_filename(filename)
    target = unique_path(config.downloads_dir, filename)

    try:
        def _progress(received, total):
            if progress_cb and total:
                try:
                    pct = received / total * 100
                    asyncio.get_event_loop().create_task(
                        progress_cb(pct, received, total)
                    )
                except Exception:  # noqa: BLE001
                    pass

        result = await mtproto_call_with_retry(
            client.download_media, message, file=str(target),
            progress_callback=_progress,
        )
    except Exception as exc:  # noqa: BLE001
        raise MTProtoError(f"Download failed: {exc}") from exc

    if not result:
        raise MTProtoError("Download produced no file.")
    path = Path(result)
    size = path.stat().st_size if path.exists() else 0
    return path, path.name, media_type, size


def _mtproto_derive_filename(message, media_type: str) -> str:
    """Best-effort original filename from a Telethon message."""
    from telethon.tl.types import DocumentAttributeFilename
    media = message.media
    if media is None:
        return f"media_{message.id}"
    doc = getattr(media, "document", None)
    if doc:
        for attr in getattr(doc, "attributes", []) or []:
            if isinstance(attr, DocumentAttributeFilename) and attr.file_name:
                return attr.file_name
        ext = {"video": ".mp4", "video_note": ".mp4", "voice": ".ogg",
               "audio": ".mp3", "document": ".bin", "photo": ".jpg"}.get(media_type, ".bin")
        return f"{getattr(doc, 'id', 'media')}{ext}"
    if getattr(media, "photo", None):
        return f"photo_{getattr(media.photo, 'id', message.id)}.jpg"
    return f"media_{message.id}"


def _mtproto_extract_single_message(result):
    """Normalise a Telethon get_messages(ids=...) result to one Message.

    Telethon returns a single Message object when ids=<int>, a list when
    ids=<list>, or None if not found. This helper returns the first non-None
    Message or None. It NEVER subscripts a single Message object.
    """
    if result is None:
        return None
    if isinstance(result, (list, tuple)):
        for item in result:
            if item is not None:
                return item
        return None
    return result


async def mtproto_screenshot(chat_ref: str | int) -> Path | None:
    """Take a text-rendered 'screenshot' of a chat's recent messages."""
    return await mtproto_take_screenshot(chat_ref)


async def mtproto_download_channel_media(chat_ref: str | int, limit: int = 50,
                                         progress_cb=None) -> list[dict]:
    """Download up to *limit* recent media messages from a channel."""
    client = _mtproto_require_client()

    try:
        entity = await mtproto_call_with_retry(client.get_entity, chat_ref)
        messages = await mtproto_call_with_retry(
            client.get_messages, entity, limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        raise MTProtoError(f"Failed to read channel: {exc}") from exc

    results = []
    for m in messages:
        if not m.media:
            continue
        try:
            path, name, mtype, size = await mtproto_download_message_media(
                chat_ref, m.id, progress_cb=progress_cb,
            )
            results.append({
                "path": str(path), "filename": name,
                "media_type": mtype, "size": size, "message_id": m.id,
            })
            if progress_cb:
                try:
                    await progress_cb(len(results), limit)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to download message #%s: %s", m.id, exc)
    return results


# ===========================================================================
# Services — media_browser (channel category browsing + Download All)
# ===========================================================================
# Browse and bulk-download media by type via MTProto. When a user enters a
# @username or channel link (without a message id) in the Download menu and
# MTProto is available, the bot scans the channel's recent history and shows
# media categories (Photos/Videos/Documents/Audio/Voice/Stickers) with item
# counts. Tapping a category lists individual items + a "Download All" button.
#
# All MTProto operations go through ``mtproto_service`` / ``mtproto_manager``.
# The Bot API remains in charge of all user interaction.
# ===========================================================================

# Map from MTProto media kind -> UI category label.
MB_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # (key, label, telethon-media-kinds)
    ("photo",     "📷 Photos",        ("photo",)),
    ("video",     "🎬 Videos",        ("video", "gif", "video_note")),
    ("document",  "📄 Documents",     ("document",)),
    ("audio",     "🎵 Audio",         ("audio",)),
    ("voice",     "🎤 Voice",         ("voice",)),
    ("sticker",   "🏷️ Stickers",      ("sticker",)),
)

MB_CATEGORY_KEYS = tuple(c[0] for c in MB_CATEGORIES)


def mb_category_label(key: str) -> str:
    for k, label, _ in MB_CATEGORIES:
        if k == key:
            return label
    return key


def mb_kind_to_category(media_kind: str) -> Optional[str]:
    """Map a media kind (from mtproto_service._media_type) to a category key."""
    for cat_key, _, kinds in MB_CATEGORIES:
        if media_kind in kinds:
            return cat_key
    return None


async def mb_scan_channel_media(chat_ref: str | int,
                                limit: int = 100) -> dict[str, list[dict]]:
    """Scan a channel's recent messages and return media grouped by category.

    Returns ``{category_key: [{message_id, media_type, date, caption, size}, …]}``.
    Only messages with media are included. Text-only messages are skipped.
    """
    history = await mtproto_service.get_channel_history(chat_ref, limit=limit)
    by_cat: dict[str, list[dict]] = {k: [] for k in MB_CATEGORY_KEYS}

    for m in history:
        if not m.get("has_media"):
            continue
        media_kind = m.get("media_type", "")
        cat = mb_kind_to_category(media_kind)
        if cat is None:
            # Uncategorised media → put under 'document' as a fallback.
            cat = "document"
        by_cat[cat].append({
            "message_id": m.get("id"),
            "media_type": media_kind,
            "date": m.get("date"),
            "text": m.get("text", ""),
            "sender_id": m.get("sender_id"),
        })
    # Remove empty categories.
    return {k: v for k, v in by_cat.items() if v}


async def mb_download_one(chat_ref: str | int, message_id: int,
                          progress_cb=None) -> dict[str, Any]:
    """Download a single media item. Returns a result dict.

    ``{ok: bool, path?: Path, filename?: str, media_type?: str, size?: int,
       error?: str, message_id: int}``
    """
    try:
        path, name, mtype, size = await mtproto_service.download_message_media(
            chat_ref, message_id, progress_cb=progress_cb,
        )
        return {
            "ok": True,
            "path": path,
            "filename": name,
            "media_type": mtype,
            "size": size,
            "message_id": message_id,
        }
    except mtproto_service.MTProtoError as exc:
        return {"ok": False, "error": str(exc), "message_id": message_id}
    except Exception as exc:  # noqa: BLE001
        logger.warning("download_one failed for #%s: %s", message_id, exc)
        return {"ok": False, "error": str(exc), "message_id": message_id}


async def mb_download_category(
    chat_ref: str | int,
    category: str,
    items: list[dict],
    *,
    progress_cb=None,
    item_done_cb=None,
    dedup: set[int] | None = None,
) -> dict[str, Any]:
    """Download all items in a category list.

    Continues on per-item failure (records the error). Deduplicates by
    message_id when a ``dedup`` set is provided (mutated in place).

    Returns ``{total, succeeded, failed, results: [...], skipped}``.
    """
    dedup = dedup if dedup is not None else set()
    results: list[dict] = []
    succeeded = 0
    failed = 0
    skipped = 0
    total = len(items)

    for i, item in enumerate(items, 1):
        msg_id = item.get("message_id")
        if msg_id is None:
            continue
        if msg_id in dedup:
            skipped += 1
            results.append({"ok": False, "skipped": True, "message_id": msg_id})
            continue
        dedup.add(msg_id)

        result = await mb_download_one(chat_ref, msg_id, progress_cb=progress_cb)
        results.append(result)
        if result.get("ok"):
            succeeded += 1
        else:
            failed += 1

        if progress_cb:
            try:
                await progress_cb(i, total, msg_id, result.get("ok", False))
            except Exception:  # noqa: BLE001
                pass
        if item_done_cb:
            try:
                await item_done_cb(result)
            except Exception:  # noqa: BLE001
                pass

        # Small delay to be gentle on Telegram's rate limits.
        await asyncio.sleep(0.3)

    return {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }


def mb_scan_summary(by_cat: dict[str, list[dict]]) -> str:
    """Render a human-readable summary of found media."""
    if not by_cat:
        return "📂 No media found in the recent messages of that channel."
    total = sum(len(v) for v in by_cat.values())
    lines = [f"📂 *Found {total} media items in {len(by_cat)} categories:*\n"]
    for cat_key in MB_CATEGORY_KEYS:
        if cat_key in by_cat:
            lines.append(f"  • {mb_category_label(cat_key)}: {len(by_cat[cat_key])}")
    return "\n".join(lines)


def mb_bulk_summary(result: dict[str, Any], category: str) -> str:
    """Render a bulk-download completion summary."""
    label = mb_category_label(category)
    return (
        f"✅ *Bulk download complete — {label}*\n\n"
        f"📊 Total: {result.get('total', 0)}\n"
        f"✅ Succeeded: {result.get('succeeded', 0)}\n"
        f"❌ Failed: {result.get('failed', 0)}\n"
        f"⏭️ Skipped (duplicates): {result.get('skipped', 0)}"
    )


# ===========================================================================
# Services — VC transport (Telethon group-call join/leave, optional)
# ===========================================================================
# Implements silent voice-chat participation via Telethon's native MTProto
# group-call APIs (JoinGroupCallRequest / LeaveGroupCallRequest). The userbot
# appears as a joined participant (muted) — correct for a silent tour bot.
#
# All Telethon imports are done lazily inside the functions, so the bot boots
# fine without Telethon installed. All operations go through the existing
# authenticated MTProto client owned by ``mtproto_manager``.
# ===========================================================================


class VCTransportError(Exception):
    """Raised when a VC transport operation fails."""


async def vc_detect_active_call(client, entity) -> Optional[dict[str, Any]]:
    """Check whether a group/supergroup has an active group call (voice chat).

    Returns a dict with ``call_id`` and ``access_hash`` if active, else None.
    """
    from telethon.tl.functions.channels import GetFullChannelRequest
    from telethon.tl.types import Channel
    from telethon.errors import (
        ChannelPrivateError, ChatAdminRequiredError, FloodWaitError,
        RPCError,
    )

    try:
        result = await mtproto_call_with_retry(client, GetFullChannelRequest, entity)
    except ChannelPrivateError:
        raise VCTransportError("Channel is private — userbot is not a member.")
    except ChatAdminRequiredError:
        raise VCTransportError("Admin rights required to view full channel info.")
    except FloodWaitError as exc:
        raise VCTransportError(f"Rate limited: must wait {exc.seconds}s.") from exc
    except RPCError as exc:
        raise VCTransportError(f"Telegram RPC error: {exc}") from exc

    full_chat = result.full_chat
    call = getattr(full_chat, "call", None)
    if call is None:
        return None  # no active call

    call_id = getattr(call, "id", None)
    access_hash = getattr(call, "access_hash", None)
    if call_id is None or access_hash is None:
        return None

    members = getattr(full_chat, "participants_count", None)
    return {
        "call_id": call_id,
        "access_hash": access_hash,
        "members": members,
    }


async def vc_join_call(client, entity, call_id: int, access_hash: int,
                       muted: bool = True) -> bool:
    """Join the active group call (voice chat).

    Returns True on success. The userbot becomes a participant (muted by
    default — silent presence).
    """
    from telethon.tl.functions.phone import JoinGroupCallRequest
    from telethon.tl.types import InputGroupCall
    from telethon.errors import (
        FloodWaitError, RPCError, BadRequestError,
    )

    call_input = InputGroupCall(id=call_id, access_hash=access_hash)
    try:
        await mtproto_call_with_retry(
            client, JoinGroupCallRequest,
            call=call_input,
            join_as=None,  # join as self
            params=None,   # no UDP transport params (silent presence)
            muted=muted,
            video_stopped=True,
            invite_hash=None,
        )
        logger.info("JoinGroupCall OK for call_id=%s", call_id)
        return True
    except FloodWaitError as exc:
        raise VCTransportError(f"FloodWait: must wait {exc.seconds}s.") from exc
    except BadRequestError as exc:
        msg = str(exc).lower()
        if "already joined" in msg or "already a participant" in msg:
            logger.info("Already joined call_id=%s — treating as success.", call_id)
            return True
        raise VCTransportError(f"Join rejected: {exc}") from exc
    except RPCError as exc:
        raise VCTransportError(f"Join RPC error: {exc}") from exc


async def vc_leave_call(client, call_id: int, access_hash: int) -> bool:
    """Leave the group call. Idempotent — safe to call when already left."""
    from telethon.tl.functions.phone import LeaveGroupCallRequest
    from telethon.tl.types import InputGroupCall
    from telethon.errors import (
        FloodWaitError, RPCError, BadRequestError,
    )

    call_input = InputGroupCall(id=call_id, access_hash=access_hash)
    try:
        await mtproto_call_with_retry(
            client, LeaveGroupCallRequest,
            call=call_input,
            source=0,  # our SSRC (0 = not streaming)
        )
        logger.info("LeaveGroupCall OK for call_id=%s", call_id)
        return True
    except FloodWaitError as exc:
        raise VCTransportError(f"FloodWait on leave: {exc.seconds}s.") from exc
    except BadRequestError as exc:
        msg = str(exc).lower()
        if "not a participant" in msg or "not in the call" in msg or "already left" in msg:
            logger.info("Already not in call_id=%s — treating as success.", call_id)
            return True
        raise VCTransportError(f"Leave rejected: {exc}") from exc
    except RPCError as exc:
        raise VCTransportError(f"Leave RPC error: {exc}") from exc


# ===========================================================================
# Services — VC tour (discovery, detection, sequential tour, manual override)
# ===========================================================================
# Architecture:
#   * Discovery: enumerate ``client.iter_dialogs()`` for groups the userbot is
#     already a member of (legitimate access only). Plus explicit admin-supplied
#     targets.
#   * Detection: for each group, ``GetFullChannelRequest`` -> check
#     ``full_chat.call``.
#   * Tour: sequential queue — join one active VC, stay N minutes, leave, next.
#   * Manual override: admin ``join`` command in a group pauses auto tour,
#     joins that group's VC, resumes when configured.
#   * Concurrency: ``asyncio.Lock`` for transitions, ``asyncio.Event`` for
#     pause/resume, single tour task, single active call.
#   * Recovery: tour state persisted to DB; on restart, stale state is
#     reconciled (we never assume the previous VC is still connected).
#
# All user-facing notifications go through the Bot API (via ``bot_registry``).
# The MTProto client only performs the VC operations.
# ===========================================================================


@dataclass
class VCTourState:
    """In-memory tour state (mirrored to DB for recovery)."""
    running: bool = False
    paused: bool = False
    current_group_id: Optional[int] = None
    current_call_id: Optional[int] = None
    current_access_hash: Optional[int] = None
    current_title: Optional[str] = None
    current_username: Optional[str] = None
    current_link: Optional[str] = None
    current_members: Optional[int] = None
    joined_at: Optional[float] = None  # monotonic
    joined_at_iso: Optional[str] = None
    planned_stay: int = 300  # seconds
    queue: list[dict] = field(default_factory=list)
    queue_index: int = 0
    mode: str = "auto"  # auto | manual
    visited_this_tour: set[int] = field(default_factory=set)


_vc_state = VCTourState()
_vc_lock = asyncio.Lock()           # guards VC transitions (join/leave)
_vc_pause_event = asyncio.Event()   # set = running, cleared = paused
_vc_pause_event.set()               # not paused by default
_vc_tour_task: Optional[asyncio.Task] = None
_vc_stop_requested = False
_vc_cached_userbot_name: str = ""
_vc_command_handler_registered = False


def _vc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _vc_now_local() -> str:
    return datetime.now().strftime("%I:%M %p")


def _vc_duration_str(seconds: float) -> str:
    s = int(seconds)
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


def vc_is_authorized(sender_id: int) -> bool:
    """True if *sender_id* is authorized to control the VC tour."""
    return sender_id in config.vc_admin_ids


# --- Group discovery -------------------------------------------------------

async def vc_discover_groups(explicit_targets: list[str] | None = None) -> dict:
    """Discover eligible groups from the userbot's dialogs + explicit targets.

    Returns a summary dict with counts.
    """
    client = mtproto_get_client()
    if client is None:
        return {"error": "MTProto not started"}

    from telethon.tl.types import Chat, Channel, Dialog
    from telethon.errors import RPCError, FloodWaitError

    discovered = 0
    skipped = 0
    errors = 0
    seen_ids: set[int] = set()
    limit = config.vc_discovery_limit

    # Source A: dialogs (groups the userbot is already a member of).
    try:
        async for dialog in client.iter_dialogs(limit=limit):
            entity = dialog.entity
            # Only groups/supergroups, not private chats or broadcast channels.
            is_group = False
            if isinstance(entity, Channel):
                # Supergroup (megagroup) yes; broadcast channel only if it has
                # a discussion group. We accept supergroups.
                if not getattr(entity, "broadcast", False):
                    is_group = True
                # Broadcast channels: skip unless they have a linked discussion
                # group (we can't easily check here without extra calls).
            elif isinstance(entity, Chat):
                is_group = True
            if not is_group:
                continue

            gid = entity.id
            if gid in seen_ids:
                continue
            seen_ids.add(gid)

            title = getattr(entity, "title", None) or "Untitled"
            username = getattr(entity, "username", None)
            link = f"https://t.me/{username}" if username else None
            access_hash = getattr(entity, "access_hash", None)

            try:
                await vc_upsert_group(
                    group_id=gid, access_hash=access_hash, title=title,
                    username=username, public_link=link, source="dialog",
                    access_status="accessible", active_vc=False,
                )
                discovered += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("vc_upsert_group failed for %s: %s", gid, exc)
                errors += 1
    except FloodWaitError as exc:
        logger.warning("FloodWait during dialog discovery: %ss", exc.seconds)
        await asyncio.sleep(min(exc.seconds + 1, 60))
    except Exception as exc:  # noqa: BLE001
        logger.exception("dialog discovery error: %s", exc)
        errors += 1

    # Source B: explicit targets supplied by admin.
    for target in (explicit_targets or []):
        try:
            entity = await client.get_entity(target)
            if entity is None:
                continue
            gid = entity.id
            if gid in seen_ids:
                continue
            seen_ids.add(gid)
            title = getattr(entity, "title", None) or str(target)
            username = getattr(entity, "username", None)
            link = f"https://t.me/{username}" if username else None
            access_hash = getattr(entity, "access_hash", None)
            await vc_upsert_group(
                group_id=gid, access_hash=access_hash, title=title,
                username=username, public_link=link, source="explicit",
                access_status="accessible", active_vc=False,
            )
            discovered += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("explicit target %s failed: %s", target, exc)
            skipped += 1

    logger.info("VC discovery: %d discovered, %d skipped, %d errors",
                discovered, skipped, errors)
    return {"discovered": discovered, "skipped": skipped, "errors": errors}


# --- Active VC detection + queue building ---------------------------------

async def vc_build_active_queue() -> list[dict]:
    """Scan all discovered groups for active VCs and return the eligible queue.

    Returns a list of dicts: ``{group_id, title, username, link, call_id,
    access_hash, members}``.
    """
    client = mtproto_get_client()
    if client is None:
        return []

    groups = await vc_groups_all(limit=500)
    queue: list[dict] = []
    for g in groups:
        gid = g["group_id"]
        try:
            entity = await client.get_entity(gid)
            info = await vc_detect_active_call(client, entity)
            if info is None:
                await vc_upsert_group(
                    group_id=gid, title=g.get("title"), username=g.get("username"),
                    public_link=g.get("public_link"), source=g.get("source"),
                    access_status="accessible", active_vc=False,
                )
                continue
            await vc_upsert_group(
                group_id=gid, title=g.get("title"), username=g.get("username"),
                public_link=g.get("public_link"), source=g.get("source"),
                access_status="accessible", active_vc=True,
            )
            queue.append({
                "group_id": gid,
                "title": g.get("title") or str(gid),
                "username": g.get("username"),
                "link": g.get("public_link"),
                "call_id": info["call_id"],
                "access_hash": info["access_hash"],
                "members": info.get("members"),
            })
            logger.info("Active VC found: %s (%s)", g.get("title"), gid)
        except VCTransportError as exc:
            await vc_upsert_group(
                group_id=gid, active_vc=False,
                access_status="inaccessible", last_error=str(exc),
            )
            logger.info("Group %s skipped: %s", gid, exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("VC check failed for %s: %s", gid, exc)
    return queue


# --- Admin notifications (via Bot API) -------------------------------------

async def _vc_notify_admin(text: str) -> None:
    """Send a VC update to the admin via the Bot API."""
    admin_id = config.mtproto_admin_id
    if not admin_id:
        return
    bot = get_bot()
    if bot is None:
        logger.warning("Bot not registered — cannot send VC admin update.")
        return
    await safe_send_message(bot, admin_id, text)


async def _vc_send_join_report(group: dict, queue_pos: int, queue_total: int,
                               mode: str) -> None:
    if not config.vc_join_notifications:
        return
    title = md_escape(group.get("title", "Unknown"))
    gid = group.get("group_id", "?")
    link = group.get("link")
    link_s = link if link else "Private / no shareable link available"
    members = group.get("members")
    members_s = f"{members:,}" if members else "Unavailable"
    me = _vc_get_userbot_username()
    text = (
        "🎙️ *VC JOINED*\n\n"
        f"🏷 *Group:* {title}\n"
        f"🆔 *Group ID:* `{gid}`\n"
        f"🔗 *Link:* {md_escape(link_s)}\n"
        f"👥 *Members:* {members_s}\n"
        f"🕒 *Joined At:* {_vc_now_local()}\n"
        f"⏳ *Planned Stay:* {config.vc_stay_minutes} minutes\n"
        f"📍 *Queue Position:* {queue_pos}/{queue_total}\n"
        f"👤 *Userbot:* @{md_escape(me)}\n"
        f"🎯 *Mode:* {mode}\n"
        f"📡 *Status:* Connected"
    )
    await _vc_notify_admin(text)


async def _vc_send_leave_report(group: dict, joined_iso: str,
                                actual_duration: float, mode: str,
                                reason: str = "Auto") -> None:
    if not config.vc_leave_notifications:
        return
    title = md_escape(group.get("title", "Unknown"))
    gid = group.get("group_id", "?")
    link = group.get("link")
    link_s = link if link else "Private / no shareable link available"
    text = (
        "🚪 *VC LEFT*\n\n"
        f"🏷 *Group:* {title}\n"
        f"🆔 *Group ID:* `{gid}`\n"
        f"🔗 *Link:* {md_escape(link_s)}\n"
        f"🕒 *Joined At:* {joined_iso}\n"
        f"🕒 *Left At:* {_vc_now_local()}\n"
        f"⏱ *Actual Duration:* {_vc_duration_str(actual_duration)}\n"
        f"🎯 *Mode:* {mode}\n"
        f"➡️ *Next:* {'Moving to next eligible active VC' if reason == 'Auto' else md_escape(reason)}"
    )
    await _vc_notify_admin(text)


def _vc_get_userbot_username() -> str:
    """Best-effort fetch of the userbot's username (cached)."""
    global _vc_cached_userbot_name
    if _vc_cached_userbot_name:
        return _vc_cached_userbot_name
    client = mtproto_get_client()
    if client is None:
        return "userbot"
    # We can't await here (sync function); return placeholder.
    return "userbot"


async def _vc_cache_userbot_info() -> None:
    global _vc_cached_userbot_name
    client = mtproto_get_client()
    if client is None:
        return
    try:
        me = await client.get_me()
        _vc_cached_userbot_name = getattr(me, "username", "") or str(me.id)
    except Exception:  # noqa: BLE001
        _vc_cached_userbot_name = "userbot"


# --- VC join / leave (with lock) ------------------------------------------

async def _vc_join(group: dict, mode: str = "auto") -> bool:
    """Join a group's active VC. Returns True on success."""
    client = mtproto_get_client()
    if client is None:
        return False

    async with _vc_lock:
        # If already in a VC, leave it first.
        if _vc_state.current_call_id is not None:
            await _vc_leave_unlocked(reason="Moving to next VC")

        gid = group["group_id"]
        call_id = group["call_id"]
        access_hash = group["access_hash"]
        try:
            entity = await client.get_entity(gid)
            await vc_join_call(client, entity, call_id, access_hash, muted=True)
        except VCTransportError as exc:
            logger.warning("VC join failed for %s: %s", gid, exc)
            if config.vc_save_history:
                await vc_add_visit(
                    group_id=gid, group_title=group.get("title"),
                    username=group.get("username"), group_link=group.get("link"),
                    joined_at=_vc_now_iso(), mode=mode, status="failed",
                    error=str(exc),
                )
            return False

        _vc_state.current_group_id = gid
        _vc_state.current_call_id = call_id
        _vc_state.current_access_hash = access_hash
        _vc_state.current_title = group.get("title")
        _vc_state.current_username = group.get("username")
        _vc_state.current_link = group.get("link")
        _vc_state.current_members = group.get("members")
        _vc_state.joined_at = time.monotonic()
        _vc_state.joined_at_iso = _vc_now_iso()
        _vc_state.mode = mode
        _vc_state.visited_this_tour.add(gid)

        # Persist tour state.
        await vc_save_tour_state(
            current_group_id=gid, running=1, paused=0 if mode == "auto" else 1,
        )
        # Record visit.
        if config.vc_save_history:
            await vc_add_visit(
                group_id=gid, group_title=group.get("title"),
                username=group.get("username"), group_link=group.get("link"),
                joined_at=_vc_state.joined_at_iso,
                planned_duration_seconds=config.vc_stay_minutes * 60,
                mode=mode, status="joined",
            )

        logger.info("VC joined: %s (%s), mode=%s", group.get("title"), gid, mode)
        await _vc_send_join_report(group, _vc_state.queue_index + 1,
                                   len(_vc_state.queue), mode)
        return True


async def _vc_leave_unlocked(reason: str = "Auto") -> None:
    """Leave the current VC (no lock — caller must hold _vc_lock)."""
    if _vc_state.current_call_id is None:
        return

    client = mtproto_get_client()
    call_id = _vc_state.current_call_id
    access_hash = _vc_state.current_access_hash or 0
    gid = _vc_state.current_group_id
    title = _vc_state.current_title
    username = _vc_state.current_username
    link = _vc_state.current_link
    joined_iso = _vc_state.joined_at_iso or _vc_now_iso()
    duration = (time.monotonic() - _vc_state.joined_at) if _vc_state.joined_at else 0

    try:
        if client is not None:
            await vc_leave_call(client, call_id, access_hash)
    except VCTransportError as exc:
        logger.warning("VC leave failed for %s: %s", gid, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("VC leave error: %s", exc)

    # Record completion.
    if config.vc_save_history:
        visits = await vc_recent_visits(limit=1)
        if visits and visits[0].get("group_id") == gid and visits[0].get("status") == "joined":
            await vc_update_visit(
                visits[0]["id"],
                left_at=_vc_now_iso(),
                actual_duration_seconds=int(duration),
                status="completed",
                leave_reason=reason,
            )
        else:
            await vc_add_visit(
                group_id=gid, group_title=title, username=username,
                group_link=link, joined_at=joined_iso, left_at=_vc_now_iso(),
                actual_duration_seconds=int(duration),
                mode=_vc_state.mode, status="completed", leave_reason=reason,
            )

    await _vc_send_leave_report(
        {"group_id": gid, "title": title, "link": link},
        joined_iso, duration, _vc_state.mode, reason,
    )

    # Update group last_joined.
    await vc_upsert_group(
        group_id=gid, title=title, username=username, public_link=link,
        source="dialog", access_status="accessible", active_vc=True,
        last_error=None,
    )

    _vc_state.current_group_id = None
    _vc_state.current_call_id = None
    _vc_state.current_access_hash = None
    _vc_state.current_title = None
    _vc_state.current_username = None
    _vc_state.current_link = None
    _vc_state.current_members = None
    _vc_state.joined_at = None
    _vc_state.joined_at_iso = None

    logger.info("VC left: %s (%s), duration=%s, reason=%s",
                title, gid, _vc_duration_str(duration), reason)


async def vc_leave_current(reason: str = "Manual Command") -> bool:
    """Public: leave the current VC (acquires lock)."""
    async with _vc_lock:
        if _vc_state.current_call_id is None:
            return False
        await _vc_leave_unlocked(reason=reason)
        return True


# --- Tour loop -------------------------------------------------------------

async def _vc_tour_loop() -> None:
    """The sequential VC tour loop. Runs as a single background task."""
    global _vc_stop_requested
    logger.info("VC tour loop started.")
    await _vc_cache_userbot_info()

    try:
        while not _vc_stop_requested:
            # Build/refresh the queue if empty.
            if not _vc_state.queue:
                logger.info("VC tour: building active queue...")
                _vc_state.queue = await vc_build_active_queue()
                _vc_state.queue_index = 0
                await vc_save_tour_state(
                    queue_json=json.dumps(_vc_state.queue),
                    current_queue_index=0,
                )
                if not _vc_state.queue:
                    logger.info("VC tour: no active VCs found. Waiting 120s...")
                    await _vc_sleep_cancellable(120)
                    continue

            # Advance through the queue.
            while _vc_state.queue_index < len(_vc_state.queue) and not _vc_stop_requested:
                # Pause check.
                await _vc_pause_event.wait()
                if _vc_stop_requested:
                    break

                group = _vc_state.queue[_vc_state.queue_index]
                gid = group["group_id"]

                # Revisit dedup.
                if not config.vc_revisit_same_group and gid in _vc_state.visited_this_tour:
                    _vc_state.queue_index += 1
                    await vc_save_tour_state(current_queue_index=_vc_state.queue_index)
                    continue

                # Join.
                joined = await _vc_join(group, mode="auto")
                if not joined:
                    _vc_state.queue_index += 1
                    await vc_save_tour_state(current_queue_index=_vc_state.queue_index)
                    await _vc_sleep_cancellable(config.vc_cooldown_seconds)
                    continue

                # Stay for configured duration (cancellable + pause-aware).
                stay_seconds = config.vc_stay_minutes * 60
                logger.info("VC tour: staying in %s for %ss",
                            group.get("title"), stay_seconds)
                await _vc_sleep_cancellable(stay_seconds)

                if _vc_stop_requested:
                    break

                # Pause check after stay.
                await _vc_pause_event.wait()

                # Leave.
                async with _vc_lock:
                    if _vc_state.current_call_id is not None:
                        await _vc_leave_unlocked(reason="Auto")

                _vc_state.queue_index += 1
                await vc_save_tour_state(current_queue_index=_vc_state.queue_index)

                # Cooldown between groups.
                if _vc_state.queue_index < len(_vc_state.queue) and not _vc_stop_requested:
                    await _vc_sleep_cancellable(config.vc_cooldown_seconds)

            # Queue exhausted — wait then loop to rebuild.
            if _vc_state.queue_index >= len(_vc_state.queue) and not _vc_stop_requested:
                logger.info("VC tour: queue exhausted. Waiting 300s before refresh.")
                _vc_state.queue = []
                _vc_state.queue_index = 0
                await vc_save_tour_state(
                    queue_json="[]", current_queue_index=0,
                )
                await _vc_sleep_cancellable(300)

    except asyncio.CancelledError:
        logger.info("VC tour loop cancelled.")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("VC tour loop error: %s", exc)
    finally:
        # Clean up: leave any active VC.
        async with _vc_lock:
            if _vc_state.current_call_id is not None:
                await _vc_leave_unlocked(reason="Tour stopped")
        _vc_state.running = False
        await vc_save_tour_state(running=0, paused=0, current_group_id=None)
        logger.info("VC tour loop ended.")


async def _vc_sleep_cancellable(seconds: float) -> None:
    """Sleep that respects pause + stop, checking every second."""
    global _vc_stop_requested
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if _vc_stop_requested:
            return
        if not _vc_pause_event.is_set():
            await _vc_pause_event.wait()  # block until resumed
        try:
            await asyncio.sleep(min(1.0, end - time.monotonic()))
        except asyncio.CancelledError:
            raise


# --- Public tour controls --------------------------------------------------

async def vc_start_tour() -> str:
    """Start the VC tour. Returns a status string."""
    global _vc_tour_task, _vc_stop_requested
    async with _vc_lock:
        if _vc_state.running and _vc_tour_task and not _vc_tour_task.done():
            return "already_running"
        _vc_stop_requested = False
        _vc_pause_event.set()
        _vc_state.running = True
        _vc_state.paused = False
        _vc_state.queue = []
        _vc_state.queue_index = 0
        _vc_state.visited_this_tour.clear()
        await vc_save_tour_state(
            running=1, paused=0, started_at=_vc_now_iso(),
            current_group_id=None, current_queue_index=0, queue_json="[]",
        )
        _vc_tour_task = asyncio.create_task(_vc_tour_loop())
        logger.info("VC tour started.")
        return "started"


async def vc_pause_tour() -> str:
    """Pause the tour (finishes current stay, then blocks)."""
    if not _vc_state.running:
        return "not_running"
    _vc_pause_event.clear()
    _vc_state.paused = True
    await vc_save_tour_state(paused=1)
    logger.info("VC tour paused.")
    return "paused"


async def vc_resume_tour() -> str:
    """Resume a paused tour."""
    if not _vc_state.running:
        return "not_running"
    _vc_pause_event.set()
    _vc_state.paused = False
    await vc_save_tour_state(paused=0)
    logger.info("VC tour resumed.")
    return "resumed"


async def vc_stop_tour() -> str:
    """Stop the tour, leave any active VC, cancel the task."""
    global _vc_stop_requested, _vc_tour_task
    _vc_stop_requested = True
    _vc_pause_event.set()  # unblock any paused wait
    if _vc_tour_task and not _vc_tour_task.done():
        _vc_tour_task.cancel()
        try:
            await asyncio.wait_for(_vc_tour_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
    _vc_tour_task = None
    # Leave active VC.
    async with _vc_lock:
        if _vc_state.current_call_id is not None:
            await _vc_leave_unlocked(reason="Tour stopped")
    _vc_state.running = False
    _vc_state.paused = False
    _vc_stop_requested = False
    await vc_save_tour_state(running=0, paused=0, current_group_id=None)
    logger.info("VC tour stopped.")
    return "stopped"


async def vc_manual_join(group_ref: str) -> str:
    """Manual override: pause auto tour, join a specific group's VC.

    *group_ref* is a @username, link, or numeric id.
    """
    client = mtproto_get_client()
    if client is None:
        return "mtproto_not_started"

    # Pause auto tour if running.
    if _vc_state.running and not _vc_state.paused:
        _vc_pause_event.clear()
        _vc_state.paused = True
        await vc_save_tour_state(paused=1)
        logger.info("Auto tour paused for manual override.")

    # Leave current VC if any.
    async with _vc_lock:
        if _vc_state.current_call_id is not None:
            await _vc_leave_unlocked(reason="Manual override")

    # Resolve + detect + join.
    try:
        entity = await client.get_entity(group_ref)
    except Exception as exc:  # noqa: BLE001
        return f"resolve_failed:{exc}"

    try:
        info = await vc_detect_active_call(client, entity)
    except VCTransportError as exc:
        return f"detect_failed:{exc}"

    if info is None:
        return "no_active_vc"

    gid = entity.id
    title = getattr(entity, "title", None) or str(group_ref)
    username = getattr(entity, "username", None)
    link = f"https://t.me/{username}" if username else None
    group = {
        "group_id": gid, "title": title, "username": username,
        "link": link, "call_id": info["call_id"],
        "access_hash": info["access_hash"], "members": info.get("members"),
    }
    joined = await _vc_join(group, mode="manual")
    if not joined:
        return "join_failed"
    return f"joined:{title}"


async def vc_manual_leave() -> str:
    """Manual leave. Auto-resumes tour if configured."""
    left = await vc_leave_current(reason="Manual Command")
    if not left:
        return "not_in_vc"
    if _vc_state.running and _vc_state.paused and config.vc_auto_resume_after_manual:
        _vc_pause_event.set()
        _vc_state.paused = False
        await vc_save_tour_state(paused=0)
        logger.info("Auto tour resumed after manual VC.")
        return "left_and_resumed"
    return "left"


async def vc_set_stay_duration(minutes: int) -> str:
    """Set the stay duration (validated against min/max).

    The config is a frozen dataclass, so the override is persisted in the
    tour-state table. The tour loop reads it from there.
    """
    if minutes < config.vc_min_stay_minutes:
        return f"too_short:min={config.vc_min_stay_minutes}"
    if minutes > config.vc_max_stay_minutes:
        return f"too_long:max={config.vc_max_stay_minutes}"
    await vc_save_tour_state(stay_seconds=minutes * 60)
    return f"set:{minutes}"


def vc_get_status() -> dict[str, Any]:
    """Return a status snapshot for the admin panel."""
    elapsed = None
    if _vc_state.joined_at and _vc_state.current_call_id is not None:
        elapsed = time.monotonic() - _vc_state.joined_at
    return {
        "running": _vc_state.running,
        "paused": _vc_state.paused,
        "current_group_id": _vc_state.current_group_id,
        "current_title": _vc_state.current_title,
        "current_username": _vc_state.current_username,
        "current_link": _vc_state.current_link,
        "current_members": _vc_state.current_members,
        "joined_at_iso": _vc_state.joined_at_iso,
        "elapsed_seconds": elapsed,
        "planned_stay": config.vc_stay_minutes * 60,
        "queue_size": len(_vc_state.queue),
        "queue_index": _vc_state.queue_index,
        "mode": _vc_state.mode,
        "visited_count": len(_vc_state.visited_this_tour),
    }


# --- Restart recovery ------------------------------------------------------

async def vc_reconcile_on_startup() -> None:
    """On startup, reconcile stale tour state. Never auto-resume blindly."""
    state = await vc_get_tour_state()
    if not state:
        return
    running = bool(state.get("running"))
    if running:
        # Previous run was interrupted. We do NOT assume the VC is still
        # connected. Mark as stopped; admin must restart manually.
        logger.info("VC tour: stale 'running' state detected — clearing.")
        await vc_save_tour_state(running=0, paused=0, current_group_id=None)
    _vc_state.running = False
    _vc_state.paused = False
    _vc_state.current_group_id = None
    _vc_state.current_call_id = None
    _vc_state.current_access_hash = None
    _vc_state.queue = []
    _vc_state.queue_index = 0
    logger.info("VC tour: state reconciled on startup.")


# --- In-group text command handler (registered with MTProto) ----------------

async def vc_register_command_handler() -> None:
    """Register the MTProto NewMessage handler for in-group VC commands."""
    global _vc_command_handler_registered
    if _vc_command_handler_registered:
        return
    client = mtproto_get_client()
    if client is None:
        return
    from telethon import events

    @client.on(events.NewMessage(incoming=True))
    async def _on_vc_command(event):  # type: ignore[no-untyped-def]
        await _vc_handle_vc_command(event)

    _vc_command_handler_registered = True
    logger.info("VC command handler registered.")


async def _vc_handle_vc_command(event) -> None:
    """Handle join/leave/status/stay commands sent in groups."""
    text = (event.raw_text or "").strip()
    if not text:
        return
    # Normalise: strip leading / and lowercase.
    cmd = text.lstrip("/").lower().split()[0] if text else ""
    if cmd not in ("join", "vcjoin", "leave", "vcleave",
                   "status", "vcstatus", "vcstay"):
        return

    sender = await event.get_sender()
    sender_id = getattr(sender, "id", 0)
    if not vc_is_authorized(sender_id):
        logger.info("Unauthorized VC command from %s: %s", sender_id, cmd)
        return

    chat = await event.get_chat()
    gid = getattr(chat, "id", None)
    if gid is None:
        return

    if cmd in ("join", "vcjoin"):
        await _vc_cmd_join(event, chat, gid)
    elif cmd in ("leave", "vcleave"):
        await _vc_cmd_leave(event)
    elif cmd in ("status", "vcstatus"):
        await _vc_cmd_status(event)
    elif cmd == "vcstay":
        await _vc_cmd_stay(event, text)


async def _vc_cmd_join(event, chat, gid: int) -> None:
    """Handle join command in a group."""
    client = mtproto_get_client()
    if client is None:
        await event.reply("⚠️ MTProto not started.")
        return
    try:
        info = await vc_detect_active_call(client, chat)
    except VCTransportError as exc:
        await event.reply(f"⚠️ {exc}")
        return
    if info is None:
        await event.reply("ℹ️ No active voice chat in this group.")
        return

    title = getattr(chat, "title", str(gid))
    username = getattr(chat, "username", None)
    link = f"https://t.me/{username}" if username else None
    group = {
        "group_id": gid, "title": title, "username": username,
        "link": link, "call_id": info["call_id"],
        "access_hash": info["access_hash"], "members": info.get("members"),
    }
    # Pause auto tour if running.
    if _vc_state.running and not _vc_state.paused:
        _vc_pause_event.clear()
        _vc_state.paused = True
        await vc_save_tour_state(paused=1)
    joined = await _vc_join(group, mode="manual")
    if joined:
        await event.reply("🎙️ Joined the voice chat (Manual Command).")
    else:
        await event.reply("⚠️ Could not join the voice chat.")


async def _vc_cmd_leave(event) -> None:
    """Handle leave command."""
    left = await vc_manual_leave()
    if "left" in left:
        await event.reply("🚪 Left the voice chat.")
    else:
        await event.reply("ℹ️ Not currently in a voice chat.")


async def _vc_cmd_status(event) -> None:
    """Handle status command."""
    s = vc_get_status()
    lines = ["🎙️ *VC Status*", ""]
    if s["current_group_id"]:
        lines.append(f"🏷 Group: {s['current_title']}")
        lines.append(f"🆔 ID: `{s['current_group_id']}`")
        lines.append(f"🕒 Joined: {s['joined_at_iso']}")
        if s["elapsed_seconds"]:
            lines.append(f"⏱ Elapsed: {_vc_duration_str(s['elapsed_seconds'])}")
        lines.append(f"🎯 Mode: {s['mode']}")
    else:
        lines.append("No active VC session.")
    lines.append(f"\nTour: {'▶️ running' if s['running'] else '⏹ stopped'}"
                 f" {'(paused)' if s['paused'] else ''}")
    lines.append(f"Queue: {s['queue_index']}/{s['queue_size']}")
    await event.reply("\n".join(lines))


async def _vc_cmd_stay(event, text: str) -> None:
    """Handle /vcstay <minutes>."""
    parts = text.split()
    if len(parts) < 2:
        await event.reply("Usage: `/vcstay <minutes>`")
        return
    try:
        minutes = int(parts[1])
    except ValueError:
        await event.reply("⚠️ Minutes must be a number.")
        return
    result = await vc_set_stay_duration(minutes)
    if result.startswith("set:"):
        await event.reply(f"✅ Stay duration set to {minutes} minutes.")
    elif "too_short" in result:
        await event.reply(f"⚠️ Minimum is {config.vc_min_stay_minutes} minutes.")
    elif "too_long" in result:
        await event.reply(f"⚠️ Maximum is {config.vc_max_stay_minutes} minutes.")
    else:
        await event.reply(f"⚠️ {result}")


# ===========================================================================
# Namespace aliases (so handler code works unchanged)
# ===========================================================================

repo = SimpleNamespace(
    upsert_user=upsert_user, ensure_settings=ensure_settings,
    update_setting=update_setting, create_task=create_task,
    update_task=update_task, add_download_history=add_download_history,
    recent_downloads=recent_downloads, add_ai_history=add_ai_history,
    recent_ai_analyses=recent_ai_analyses, clear_history=clear_history,
    stats=stats, fetch_all=fetch_all,
    add_library_entry=add_library_entry, library_entries=library_entries,
    library_search=library_search, library_count=library_count,
    library_remove=library_remove, library_clear=library_clear,
    add_inspected_chat=add_inspected_chat, recent_inspected=recent_inspected,
    user_daily_download_usage=user_daily_download_usage,
    user_stats=user_stats, global_stats=global_stats,
    add_scheduled_task=add_scheduled_task,
    pending_scheduled_tasks=pending_scheduled_tasks,
    list_scheduled_tasks=list_scheduled_tasks,
    update_scheduled_task=update_scheduled_task,
    cancel_scheduled_task=cancel_scheduled_task,
    create_broadcast=create_broadcast,
    update_broadcast_counts=update_broadcast_counts,
    recent_broadcasts=recent_broadcasts,
    all_users=all_users, user_count=user_count,
    export_user_data=export_user_data,
    restore_user_settings=restore_user_settings,
    vc_upsert_group=vc_upsert_group, vc_groups_active=vc_groups_active,
    vc_groups_all=vc_groups_all, vc_add_visit=vc_add_visit,
    vc_update_visit=vc_update_visit, vc_recent_visits=vc_recent_visits,
    vc_visit_count=vc_visit_count, vc_get_tour_state=vc_get_tour_state,
    vc_save_tour_state=vc_save_tour_state,
    vc_visited_group_ids_since=vc_visited_group_ids_since,
)

kb = SimpleNamespace(
    main_menu=main_menu, download_menu=download_menu, analyze_menu=analyze_menu,
    ai_modes_menu=ai_modes_menu, inspector_menu=inspector_menu,
    toolbox_menu=toolbox_menu, library_menu=library_menu,
    library_entry_keyboard=library_entry_keyboard, stats_menu=stats_menu,
    qr_menu=qr_menu, batch_menu=batch_menu, scheduled_menu=scheduled_menu,
    backup_menu=backup_menu, admin_menu=admin_menu, history_menu=history_menu,
    settings_menu=settings_menu, help_menu=help_menu, back_only=back_only,
    cancel_back=cancel_back, settings_back=settings_back, help_back=help_back,
    history_back=history_back, library_back=library_back,
    inspector_back=inspector_back, toolbox_back=toolbox_back,
    stats_back=stats_back, qr_back=qr_back,
    download_done_keyboard=download_done_keyboard, batch_back=batch_back,
    scheduled_back=scheduled_back, backup_back=backup_back,
    admin_back=admin_back, ai_modes_back=ai_modes_back,
    admin_only_menu=admin_only_menu,
    mtproto_menu=mtproto_menu, mtproto_back=mtproto_back,
    vc_menu=vc_menu, vc_manual_menu=vc_manual_menu,
    vc_settings_menu=vc_settings_menu, vc_back=vc_back,
    vcmanual_back=vcmanual_back,
)

msg = SimpleNamespace(
    WELCOME=WELCOME, MAIN_MENU_TEXT=MAIN_MENU_TEXT,
    DOWNLOAD_MENU_TEXT=DOWNLOAD_MENU_TEXT, ANALYZE_MENU_TEXT=ANALYZE_MENU_TEXT,
    INSPECTOR_MENU_TEXT=INSPECTOR_MENU_TEXT, TOOLBOX_MENU_TEXT=TOOLBOX_MENU_TEXT,
    LIBRARY_MENU_TEXT=LIBRARY_MENU_TEXT, STATS_MENU_TEXT=STATS_MENU_TEXT,
    QR_MENU_TEXT=QR_MENU_TEXT, HISTORY_MENU_TEXT=HISTORY_MENU_TEXT,
    SETTINGS_MENU_TEXT=SETTINGS_MENU_TEXT, HELP_MENU_TEXT=HELP_MENU_TEXT,
    QUALITIES=QUALITIES, AI_MODELS=AI_MODELS, LANGUAGES=LANGUAGES,
    DOWNLOAD_FORWARD_PROMPT=DOWNLOAD_FORWARD_PROMPT,
    DOWNLOAD_CANCEL_PROMPT=DOWNLOAD_CANCEL_PROMPT,
    ANALYZE_FORWARD_PROMPT=ANALYZE_FORWARD_PROMPT,
    ANALYZE_CANCEL_PROMPT=ANALYZE_CANCEL_PROMPT,
    INSPECTOR_PROMPT=INSPECTOR_PROMPT, INSPECTOR_EMPTY=INSPECTOR_EMPTY,
    TOOLBOX_PROMPTS=TOOLBOX_PROMPTS, TOOLBOX_INVALID_MEDIA=TOOLBOX_INVALID_MEDIA,
    LIBRARY_EMPTY=LIBRARY_EMPTY, QR_PROMPT=QR_PROMPT,
    BATCH_MENU_TEXT=BATCH_MENU_TEXT, BATCH_PROMPT=BATCH_PROMPT,
    BATCH_TIMEOUT=BATCH_TIMEOUT, BATCH_EMPTY=BATCH_EMPTY,
    SCHEDULED_MENU_TEXT=SCHEDULED_MENU_TEXT, SCHEDULE_PROMPT=SCHEDULE_PROMPT,
    SCHEDULE_INVALID=SCHEDULE_INVALID, BACKUP_MENU_TEXT=BACKUP_MENU_TEXT,
    BACKUP_EXPORTING=BACKUP_EXPORTING, BACKUP_IMPORT_PROMPT=BACKUP_IMPORT_PROMPT,
    ADMIN_DENIED=ADMIN_DENIED, ADMIN_MENU_TEXT=ADMIN_MENU_TEXT,
    ADMIN_BCAST_PROMPT=ADMIN_BCAST_PROMPT,
    HELP_DOWNLOAD=HELP_DOWNLOAD, HELP_AI=HELP_AI,
    HELP_FEATURES=HELP_FEATURES, HELP_FORMATS=HELP_FORMATS,
    HELP_FAQ=HELP_FAQ, AI_MODES_TEXT=AI_MODES_TEXT,
    settings_text=settings_text, progress_text=progress_text,
    download_done_text=download_done_text,
    download_failed_text=download_failed_text,
    file_too_big_text=file_too_big_text,
    analyze_result_text=analyze_result_text,
    analyze_failed_text=analyze_failed_text,
    render_history_rows=render_history_rows,
    inspect_result_text=inspect_result_text,
    inspect_failed_text=inspect_failed_text,
    media_info_text=media_info_text,
    library_entry_text=library_entry_text,
    library_list_text=library_list_text,
    user_stats_text=user_stats_text,
    global_stats_text=global_stats_text,
    quota_exceeded_text=quota_exceeded_text,
    ai_mode_set_text=ai_mode_set_text,
    batch_summary_text=batch_summary_text,
    batch_failed_text=batch_failed_text,
    scheduled_list_text=scheduled_list_text,
    backup_export_done_text=backup_export_done_text,
    backup_import_done_text=backup_import_done_text,
    backup_failed_text=backup_failed_text,
    admin_users_text=admin_users_text,
    admin_bcast_done_text=admin_bcast_done_text,
    inline_help_text=inline_help_text,
    inline_stats_text=inline_stats_text,
    LINK_DOWNLOAD_PROMPT=LINK_DOWNLOAD_PROMPT,
    LINK_DOWNLOAD_EMPTY=LINK_DOWNLOAD_EMPTY,
    link_parse_error_text=link_parse_error_text,
    link_resolving_text=link_resolving_text,
    link_download_failed_text=link_download_failed_text,
    link_resolved_ask_msgid=link_resolved_ask_msgid,
)

states = SimpleNamespace(
    IDLE=IDLE,
    AWAIT_DOWNLOAD_FORWARD=AWAIT_DOWNLOAD_FORWARD,
    AWAIT_ANALYZE=AWAIT_ANALYZE,
    AWAIT_INSPECT=AWAIT_INSPECT,
    AWAIT_TOOLBOX=AWAIT_TOOLBOX,
    AWAIT_LIBRARY_SEARCH=AWAIT_LIBRARY_SEARCH,
    AWAIT_QR=AWAIT_QR,
    AWAIT_BATCH=AWAIT_BATCH,
    AWAIT_SCHEDULE=AWAIT_SCHEDULE,
    AWAIT_BACKUP_IMPORT=AWAIT_BACKUP_IMPORT,
    AWAIT_ADMIN_BCAST=AWAIT_ADMIN_BCAST,
    AWAIT_LINK_DOWNLOAD=AWAIT_LINK_DOWNLOAD,
    AWAIT_LINK_MESSAGE_ID=AWAIT_LINK_MESSAGE_ID,
    AWAIT_MTPROTO_SCREENSHOT=AWAIT_MTPROTO_SCREENSHOT,
    AWAIT_MTPROTO_DOWNLOAD=AWAIT_MTPROTO_DOWNLOAD,
    AWAIT_MTPROTO_MSGID=AWAIT_MTPROTO_MSGID,
    AWAIT_MEDIA_BROWSE=AWAIT_MEDIA_BROWSE,
    AWAIT_VC_JOIN_TARGET=AWAIT_VC_JOIN_TARGET,
    AWAIT_VC_CHECK_TARGET=AWAIT_VC_CHECK_TARGET,
    AWAIT_VC_STAY=AWAIT_VC_STAY,
    set_state=set_state, get_state=get_state, reset=reset,
    set_tool=set_tool, get_tool=get_tool,
    set_last_download=set_last_download, get_last_download=get_last_download,
    set_state_sync=set_state_sync, get_state_sync=get_state_sync,
    set_tool_sync=set_tool_sync, get_tool_sync=get_tool_sync,
    set_last_download_sync=set_last_download_sync,
    get_last_download_sync=get_last_download_sync,
)

downloader = SimpleNamespace(
    download_forwarded=download_forwarded,
    DownloadError=DownloadError,
    FileTooLarge=FileTooLarge,
)

ai_analyzer = SimpleNamespace(
    analyze_frames=analyze_frames,
    analyze_audio=analyze_audio,
    analyze_images=analyze_images,
    cleanup_frames=cleanup_frames,
)

ai_modes = SimpleNamespace(
    ACCEPTS=ACCEPTS, MODE_LABELS=MODE_LABELS, MODE_ORDER=MODE_ORDER,
    PROMPTS=PROMPTS, build_prompt=build_prompt,
    NORMALISERS=NORMALISERS, normalise=normalise,
    accepts=accepts, media_kind_for=media_kind_for,
    render_result=render_result,
)

media_tools = SimpleNamespace(
    extract_audio=extract_audio, extract_thumbnail=extract_thumbnail,
    media_info=media_info, compress_video=compress_video,
    convert_image=convert_image, ToolError=ToolError,
)

qr_generator = SimpleNamespace(
    generate_qr=generate_qr, QRError=QRError,
)

backup = SimpleNamespace(
    export_user_json=export_user_json,
    parse_backup_file=parse_backup_file,
    restore_from_dict=restore_from_dict,
    export_global_json=export_global_json,
)

frame_extractor = SimpleNamespace(
    extract_frames=extract_frames,
    FrameExtractionError=FrameExtractionError,
)

media_processor = SimpleNamespace(
    download_slot=download_slot, with_retries=with_retries,
    retrying=retrying, send_file_back=send_file_back,
    notify=notify, safe_unlink=safe_unlink, human_size=human_size,
)

i18n = SimpleNamespace(
    supported_languages=supported_languages, t=t, has_language=has_language,
)

link_parser = SimpleNamespace(
    parse_input=parse_input, describe_input=describe_input,
    from_chat_reference=from_chat_reference, ParsedInput=ParsedInput,
    KIND_MESSAGE_LINK=KIND_MESSAGE_LINK, KIND_CHANNEL_LINK=KIND_CHANNEL_LINK,
    KIND_USERNAME=KIND_USERNAME, KIND_CHAT_ID=KIND_CHAT_ID,
    VIS_PUBLIC=VIS_PUBLIC, VIS_PRIVATE=VIS_PRIVATE,
)

session_manager = SimpleNamespace(
    session_path=session_path,
    session_file_exists=session_file_exists,
    validate_session=validate_session,
    interactive_login=interactive_login,
    session_info=session_info,
)

mtproto_manager = SimpleNamespace(
    is_available=mtproto_is_available,
    is_started=mtproto_is_started,
    start=mtproto_start,
    stop=mtproto_stop,
    restart=mtproto_restart,
    get_client=mtproto_get_client,
    get_status=mtproto_get_status,
    call_with_retry=mtproto_call_with_retry,
)

mtproto_service = SimpleNamespace(
    MTProtoError=MTProtoError,
    MTProtoNotAvailable=MTProtoNotAvailable,
    is_available=mtproto_service_is_available,
    resolve_entity=mtproto_resolve_entity,
    get_channel_history=mtproto_get_channel_history,
    download_message_media=mtproto_download_message_media,
    download_channel_media=mtproto_download_channel_media,
    screenshot=mtproto_screenshot,
)

media_browser = SimpleNamespace(
    CATEGORIES=MB_CATEGORIES,
    CATEGORY_KEYS=MB_CATEGORY_KEYS,
    category_label=mb_category_label,
    kind_to_category=mb_kind_to_category,
    scan_channel_media=mb_scan_channel_media,
    download_one=mb_download_one,
    download_category=mb_download_category,
    scan_summary=mb_scan_summary,
    bulk_summary=mb_bulk_summary,
)

vc_transport = SimpleNamespace(
    VCTransportError=VCTransportError,
    detect_active_call=vc_detect_active_call,
    join_call=vc_join_call,
    leave_call=vc_leave_call,
)

vc_tour = SimpleNamespace(
    discover_groups=vc_discover_groups,
    build_active_queue=vc_build_active_queue,
    start_tour=vc_start_tour,
    pause_tour=vc_pause_tour,
    resume_tour=vc_resume_tour,
    stop_tour=vc_stop_tour,
    manual_join=vc_manual_join,
    manual_leave=vc_manual_leave,
    set_stay_duration=vc_set_stay_duration,
    get_status=vc_get_status,
    reconcile_on_startup=vc_reconcile_on_startup,
    register_command_handler=vc_register_command_handler,
    is_authorized=vc_is_authorized,
    leave_current=vc_leave_current,
)


# ===========================================================================
# Handlers — download
# ===========================================================================


async def download__enter_forward_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await states.set_state(update.effective_user.id, states.AWAIT_DOWNLOAD_FORWARD)
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.DOWNLOAD_FORWARD_PROMPT,
        reply_markup=kb.cancel_back("dl"),
        parse_mode="Markdown",
    )


async def download__handle_forwarded_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    user_id = update.effective_user.id
    media_type = telegram_media_type(message)

    if media_type is None:
        await message.reply_text(
            "⚠️ That message has no downloadable media.\n"
            "Please forward a message containing a video, document, photo, "
            "audio, voice, sticker, GIF or video note.",
        )
        return

    approx = download__approx_size(message)
    if approx and approx > config.download_limit_bytes:
        await message.reply_text(
            msg.file_too_big_text(config.max_file_size_mb),
            parse_mode="Markdown",
            reply_markup=kb.download_menu(),
        )
        return

    count_used, bytes_used = await repo.user_daily_download_usage(user_id)
    if (count_used >= config.user_daily_download_limit
            or bytes_used + approx > config.user_daily_download_bytes):
        await message.reply_text(
            msg.quota_exceeded_text(config.user_daily_download_limit,
                                    config.user_daily_download_bytes_mb),
            parse_mode="Markdown",
            reply_markup=kb.main_menu(),
        )
        return

    status = await message.reply_text(
        msg.progress_text("📥 Downloading", 0.0, 0, 1),
        reply_markup=kb.cancel_back("dl"),
        parse_mode="Markdown",
    )
    editor = _DownloadProgressEditor(context, status.chat_id, status.message_id, "📥 Downloading")

    task_id = await repo.create_task(user_id, "download", "running")
    asyncio.create_task(
        download__run_forward_download(user_id, task_id, message, context, editor, media_type)
    )


async def download__run_forward_download(
    user_id: int, task_id: int, message, context, editor, media_type: str,
) -> None:
    path: Path | None = None
    file_unique_id = download__file_unique_id(message)
    try:
        async with download_slot():
            path, name, mtype, size = await downloader.download_forwarded(
                message,
                progress=editor.callback,
                bot=context.bot,
            )
        await editor.finish()
        await repo.update_task(task_id, status="done", progress=100)

        upload_editor = _DownloadProgressEditor(
            context, editor.chat_id, None, "📤 Uploading", new_message=True
        )
        caption = msg.download_done_text(name, size, mtype)
        result = await send_file_back(
            editor.chat_id, path, caption, context,
            progress_cb=upload_editor.callback,
        )
        await upload_editor.finish()

        if result == "too_large":
            await context.bot.send_message(
                chat_id=editor.chat_id,
                text=(
                    "ℹ️ The downloaded file is larger than 50 MB, which is the "
                    "maximum a bot can send back through the official Bot API."
                ),
                reply_markup=kb.main_menu(),
            )
        elif result.startswith("error"):
            await context.bot.send_message(
                chat_id=editor.chat_id,
                text=f"⚠️ Could not send the file back: `{result[6:]}`",
                parse_mode="Markdown",
                reply_markup=kb.main_menu(),
            )

        await repo.add_download_history(
            user_id, file_name=name, file_size=size,
            mime_type=None, media_type=mtype, source="forward",
            status="done", task_id=task_id, file_unique_id=file_unique_id,
        )

        await states.set_last_download(user_id, {
            "file_name": name,
            "media_type": mtype,
            "file_size": size,
            "file_id": download__file_id(message),
        })

        if result == "ok":
            try:
                await context.bot.send_message(
                    chat_id=editor.chat_id,
                    text="💾 *Tip:* tap below to bookmark this file to your Library.",
                    parse_mode="Markdown",
                    reply_markup=kb.download_done_keyboard(),
                )
            except Exception:  # noqa: BLE001
                pass

        settings = await repo.ensure_settings(user_id)
        if settings.get("auto_delete"):
            await remove_path(path)

    except downloader.FileTooLarge as exc:
        await editor.error(msg.file_too_big_text(config.max_file_size_mb))
        await repo.update_task(task_id, status="failed", error=str(exc))
        await repo.add_download_history(
            user_id, file_name="(too large)", file_size=0,
            mime_type=None, media_type=media_type, source="forward",
            status="failed", task_id=task_id, file_unique_id=file_unique_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("forward download failed: %s", exc)
        await editor.error(msg.download_failed_text(str(exc)))
        await repo.update_task(task_id, status="failed", error=str(exc))
        await repo.add_download_history(
            user_id, file_name="(failed)", file_size=0,
            mime_type=None, media_type=media_type, source="forward",
            status="failed", task_id=task_id, file_unique_id=file_unique_id,
        )
    finally:
        if path and path.exists():
            try:
                await remove_path(path)
            except Exception:  # noqa: BLE001
                pass


class _DownloadProgressEditor:
    """Throttled editor for a status message showing a progress bar."""

    def __init__(self, context, chat_id: int, message_id: int | None,
                 label: str, *, new_message: bool = False):
        self.context = context
        self.chat_id = chat_id
        self.message_id = message_id
        self.label = label
        self.new_message = new_message
        self._last_ts = 0.0
        self._last_pct = -100.0
        self._finished = False

    async def callback(self, percent: float, received: int, total: int,
                       speed: str = "") -> None:
        if self._finished:
            return
        now = time.monotonic()
        if (now - self._last_ts) < 1.0 and (percent - self._last_pct) < 5.0 \
                and percent < 99.0:
            return
        self._last_ts = now
        self._last_pct = percent
        text = msg.progress_text(self.label, percent, received, total, speed)
        try:
            if self.new_message or self.message_id is None:
                sent = await self.context.bot.send_message(
                    chat_id=self.chat_id, text=text, parse_mode="Markdown"
                )
                self.message_id = sent.message_id
                self.new_message = False
            else:
                await self.context.bot.edit_message_text(
                    chat_id=self.chat_id, message_id=self.message_id,
                    text=text, parse_mode="Markdown",
                )
        except Exception as exc:  # noqa: BLE001 — "not modified", rate limits
            logger.debug("progress edit skipped: %s", exc)

    async def finish(self) -> None:
        self._finished = True

    async def error(self, text: str) -> None:
        self._finished = True
        try:
            if self.message_id is not None:
                await self.context.bot.edit_message_text(
                    chat_id=self.chat_id, message_id=self.message_id,
                    text=text, parse_mode="Markdown",
                    reply_markup=kb.download_menu(),
                )
            else:
                await self.context.bot.send_message(
                    chat_id=self.chat_id, text=text, parse_mode="Markdown",
                    reply_markup=kb.download_menu(),
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("error edit failed: %s", exc)


def download__approx_size(message) -> int:
    for cand in (message.document, message.video, message.audio,
                 message.animation, message.voice, message.video_note):
        if cand is not None and getattr(cand, "file_size", None):
            return int(cand.file_size)
    if message.photo:
        return int(message.photo[-1].file_size or 0)
    return 0


def download__file_unique_id(message) -> str | None:
    for cand in (message.document, message.video, message.audio,
                 message.animation, message.voice, message.video_note):
        if cand is not None and getattr(cand, "file_unique_id", None):
            return cand.file_unique_id
    if message.photo:
        return message.photo[-1].file_unique_id
    return None


def download__file_id(message) -> str | None:
    """The Bot API file_id of the forwarded media (for Library bookmarks)."""
    for cand in (message.document, message.video, message.audio,
                 message.animation, message.voice, message.video_note):
        if cand is not None and getattr(cand, "file_id", None):
            return cand.file_id
    if message.photo:
        return message.photo[-1].file_id
    return None


# ===========================================================================
# Handlers — analyze
# ===========================================================================


def analyze__approx_video_size(message) -> int:
    for cand in (message.video, message.animation, message.video_note,
                 message.document, message.audio, message.voice):
        if cand is not None and getattr(cand, "file_size", None):
            return int(cand.file_size)
    if message.photo:
        return int(message.photo[-1].file_size or 0)
    return 0


async def analyze__enter_analyze_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await states.set_state(update.effective_user.id, states.AWAIT_ANALYZE)
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.ANALYZE_FORWARD_PROMPT,
        reply_markup=kb.cancel_back("ai"),
        parse_mode="Markdown",
    )


async def analyze__show_modes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s = await repo.ensure_settings(update.effective_user.id)
    current = s.get("ai_mode", "movie")
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.AI_MODES_TEXT,
        reply_markup=kb.ai_modes_menu(current),
        parse_mode="Markdown",
    )


async def analyze__set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE,
                            mode: str) -> None:
    if mode not in ai_modes.MODE_ORDER:
        await update.callback_query.answer("Unknown mode.")
        return
    await repo.update_setting(update.effective_user.id, "ai_mode", mode)
    await update.callback_query.answer(f"Mode: {MODE_LABELS.get(mode, mode)}")
    await update.callback_query.edit_message_text(
        text=msg.ai_mode_set_text(MODE_LABELS.get(mode, mode)),
        reply_markup=kb.ai_modes_menu(mode),
        parse_mode="Markdown",
    )


async def analyze__handle_forwarded_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    user_id = update.effective_user.id
    settings = await repo.ensure_settings(user_id)
    mode = settings.get("ai_mode", "movie")
    media_type = telegram_media_type(message)

    ptb_kind = analyze__ptb_kind(media_type)
    if ptb_kind is None or not ai_modes.accepts(mode, ptb_kind):
        await message.reply_text(
            f"⚠️ The *{MODE_LABELS.get(mode, mode)}* mode needs "
            f"{analyze__accepted_kinds_text(mode)}. You sent a *{media_type}*.",
            parse_mode="Markdown",
            reply_markup=kb.analyze_menu(),
        )
        return

    approx = analyze__approx_video_size(message)
    if approx and approx > config.download_limit_bytes:
        await message.reply_text(
            msg.file_too_big_text(config.max_file_size_mb),
            parse_mode="Markdown",
            reply_markup=kb.analyze_menu(),
        )
        return

    status = await message.reply_text(
        f"🎬 Starting *{mode}* analysis…",
        reply_markup=kb.cancel_back("ai"),
        parse_mode="Markdown",
    )
    task_id = await repo.create_task(user_id, "analyze", "running")
    asyncio.create_task(
        analyze__run_analysis(user_id, task_id, message, context,
                              status.chat_id, status.message_id, media_type, mode, settings)
    )


async def analyze__run_analysis(
    user_id: int, task_id: int, message, context,
    chat_id: int, status_msg_id: int, media_type: str, mode: str,
    settings: dict,
) -> None:
    src_path: Path | None = None
    frames: list[Path] = []
    try:
        ai_model = settings.get("ai_model", config.default_ai_model)
        user_key = settings.get("gemini_api_key") or None
        lang = settings.get("language")

        await analyze__edit_status(context, chat_id, status_msg_id,
                                   "📥 Downloading media for analysis…")
        async with download_slot():
            src_path, name, mtype, size = await downloader.download_forwarded(
                message, progress=None, bot=context.bot
            )

        kind = ai_modes.media_kind_for(src_path)
        await analyze__edit_status(context, chat_id, status_msg_id,
                                   f"🧠 Running *{mode}* with `{ai_model}`…")

        if mode == "transcribe" and kind == "audio":
            result = await ai_analyzer.analyze_audio(
                src_path, model=ai_model, mode=mode,
                user_api_key=user_key, target_language=lang,
            )
        elif mode == "movie" and kind == "video":
            await analyze__edit_status(context, chat_id, status_msg_id,
                                       f"🎞️ Extracting {config.num_frames} frames…")
            frames = await frame_extractor.extract_frames(src_path)
            result = await ai_analyzer.analyze_frames(
                frames, model=ai_model, mode=mode,
                user_api_key=user_key, target_language=lang,
            )
        elif mode in ("ocr", "describe", "translate") and kind == "image":
            result = await ai_analyzer.analyze_images(
                [src_path], model=ai_model, mode=mode,
                user_api_key=user_key, target_language=lang,
            )
        elif mode in ("ocr", "describe", "translate", "movie") and kind == "video":
            await analyze__edit_status(context, chat_id, status_msg_id,
                                       f"🎞️ Extracting {config.num_frames} frames…")
            frames = await frame_extractor.extract_frames(src_path)
            result = await ai_analyzer.analyze_frames(
                frames, model=ai_model, mode=mode,
                user_api_key=user_key, target_language=lang,
            )
        else:
            raise RuntimeError(
                f"Mode {mode} cannot process media kind {kind}."
            )

        text = ai_modes.render_result(mode, result)
        await analyze__edit_status(context, chat_id, status_msg_id, text,
                                   reply_markup=kb.main_menu())
        await repo.update_task(task_id, status="done", progress=100)
        await repo.add_ai_history(
            user_id, file_name=name, media_type=mtype,
            result=analyze__to_history(result, mode), status="done", task_id=task_id,
        )

    except downloader.FileTooLarge as exc:
        await analyze__edit_status(context, chat_id, status_msg_id,
                                   msg.file_too_big_text(config.max_file_size_mb),
                                   reply_markup=kb.analyze_menu())
        await repo.update_task(task_id, status="failed", error=str(exc))
        await repo.add_ai_history(
            user_id, file_name="(too large)", media_type=media_type,
            result=None, status="failed", task_id=task_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("analysis failed: %s", exc)
        await analyze__edit_status(context, chat_id, status_msg_id,
                                   msg.analyze_failed_text(str(exc)),
                                   reply_markup=kb.analyze_menu())
        await repo.update_task(task_id, status="failed", error=str(exc))
        await repo.add_ai_history(
            user_id, file_name="(failed)", media_type=media_type,
            result=None, status="failed", task_id=task_id,
        )
    finally:
        for f in frames:
            await remove_path(f)
        if src_path and src_path.exists():
            await remove_path(src_path)


def analyze__ptb_kind(media_type: str | None) -> str | None:
    """Map PTB media type to the ai_modes media kind."""
    if media_type in ("video", "gif", "video_note"):
        return "video"
    if media_type in ("audio", "voice"):
        return "audio"
    if media_type in ("photo", "sticker"):
        return "image"
    if media_type == "document":
        return "other"
    return None


def analyze__accepted_kinds_text(mode: str) -> str:
    kinds = ai_modes.ACCEPTS.get(mode, set())
    return " or ".join(kinds) or "media"


def analyze__to_history(result: dict, mode: str) -> dict:
    """Adapt a mode result to the ai_history schema (category/title/etc)."""
    if mode == "movie":
        return result
    return {
        "category": mode.title(),
        "title": (result.get("summary") or result.get("text")
                  or result.get("details") or result.get("translated_text")
                  or "")[:200],
        "confidence": result.get("confidence", 0.0),
    }


async def analyze__edit_status(context, chat_id: int, message_id: int, text: str,
                               reply_markup=None) -> None:
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=message_id,
            text=text, parse_mode="Markdown", reply_markup=reply_markup,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("status edit skipped: %s", exc)


# ===========================================================================
# Handlers — history
# ===========================================================================


async def history__show_downloads(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    rows = await repo.recent_downloads(update.effective_user.id, limit=10)
    text = msg.render_history_rows(rows, "📥 Recent Downloads")
    await update.callback_query.edit_message_text(
        text=text, reply_markup=kb.history_back(), parse_mode="Markdown",
    )


async def history__show_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    rows = await repo.recent_ai_analyses(update.effective_user.id, limit=10)
    text = msg.render_history_rows(rows, "🤖 Recent AI Analyses")
    await update.callback_query.edit_message_text(
        text=text, reply_markup=kb.history_back(), parse_mode="Markdown",
    )


async def history__clear(update: Update, context: ContextTypes.DEFAULT_TYPE,
                         data: str) -> None:
    await update.callback_query.answer()
    kind = data.split(":")[-1] if ":" in data else None  # dl|ai|all
    if kind == "dl":
        await repo.clear_history(update.effective_user.id, "download")
        text = "🧹 *Download history cleared.*"
    elif kind == "ai":
        await repo.clear_history(update.effective_user.id, "ai")
        text = "🧹 *AI analysis history cleared.*"
    else:
        await repo.clear_history(update.effective_user.id, None)
        text = "🧹 *All history cleared.*"
    logger.info("history cleared for %s (kind=%s)", update.effective_user.id, kind)
    await update.callback_query.edit_message_text(
        text=text, reply_markup=kb.history_menu(), parse_mode="Markdown",
    )


# ===========================================================================
# Handlers — settings
# ===========================================================================

_SETTINGS_QUALITIES = QUALITIES
_SETTINGS_MODELS = AI_MODELS
_SETTINGS_LANGUAGES = LANGUAGES


def _settings_cycle(current, options):
    try:
        idx = options.index(current)
    except ValueError:
        idx = -1
    return options[(idx + 1) % len(options)]


async def settings__handle_setting(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                   data: str) -> None:
    key = data.split(":", 1)[1]
    user_id = update.effective_user.id
    settings = await repo.ensure_settings(user_id)

    if key == "q":
        new_val = _settings_cycle(settings.get("preferred_quality"), _SETTINGS_QUALITIES)
        settings = await repo.update_setting(user_id, "preferred_quality", new_val)
        toast = f"Quality → {new_val}"
    elif key == "m":
        new_val = _settings_cycle(settings.get("ai_model"), _SETTINGS_MODELS)
        settings = await repo.update_setting(user_id, "ai_model", new_val)
        toast = f"AI Model → {new_val}"
    elif key == "l":
        new_val = _settings_cycle(settings.get("language"), _SETTINGS_LANGUAGES)
        settings = await repo.update_setting(user_id, "language", new_val)
        toast = f"Language → {new_val}"
    elif key == "ad":
        new_val = 0 if settings.get("auto_delete") else 1
        settings = await repo.update_setting(user_id, "auto_delete", new_val)
        toast = f"Auto Delete → {'On' if new_val else 'Off'}"
    elif key == "n":
        new_val = 0 if settings.get("notifications") else 1
        settings = await repo.update_setting(user_id, "notifications", new_val)
        toast = f"Notifications → {'On' if new_val else 'Off'}"
    elif key == "am":
        current = settings.get("ai_mode", "movie")
        new_val = _settings_cycle(current, ai_modes.MODE_ORDER)
        settings = await repo.update_setting(user_id, "ai_mode", new_val)
        toast = f"AI Mode → {ai_modes.MODE_LABELS.get(new_val, new_val)}"
    elif key == "key":
        await states.set_state(user_id, "await_gemini_key")
        await update.callback_query.answer("Send your Gemini API key now.")
        await update.callback_query.edit_message_text(
            text=(
                "🔑 *Set your Gemini API key*\n\n"
                "Send your key as the next message. Get one at "
                "https://aistudio.google.com/app/apikey\n\n"
                "Your key is stored locally and used only for your AI requests. "
                "Send `clear` to remove it and use the shared key."
            ),
            reply_markup=kb.cancel_back("set"),
            parse_mode="Markdown",
        )
        return
    else:
        toast = "Unknown setting"

    try:
        await update.callback_query.answer(toast)
    except Exception:  # noqa: BLE001
        pass
    try:
        await update.callback_query.edit_message_text(
            text=msg.settings_text(settings),
            reply_markup=kb.settings_menu(),
            parse_mode="Markdown",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("settings edit skipped: %s", exc)


async def settings__handle_gemini_key_input(update: Update,
                                            context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    if not text:
        await update.message.reply_text("⚠️ Please send your API key.")
        return
    await states.reset(user_id)
    if text.lower() == "clear":
        await repo.update_setting(user_id, "gemini_api_key", None)
        await update.message.reply_text(
            "✅ Your personal Gemini key was removed. The bot will use the "
            "shared key (if configured).",
            reply_markup=kb.settings_back(),
        )
        return
    if not text.startswith("AIza"):
        await update.message.reply_text(
            "⚠️ That doesn't look like a Gemini API key (they usually start "
            "with `AIza`). Try again or send `clear`.",
            parse_mode="Markdown",
            reply_markup=kb.settings_back(),
        )
        return
    await repo.update_setting(user_id, "gemini_api_key", text)
    await update.message.reply_text(
        "✅ Your Gemini API key has been saved. Future AI requests will use it.",
        reply_markup=kb.settings_back(),
    )


# ===========================================================================
# Handlers — help
# ===========================================================================

_HELP_SECTIONS = {
    "hp:dl": HELP_DOWNLOAD,
    "hp:ai": HELP_AI.format(n=config.num_frames),
    "hp:feat": HELP_FEATURES,
    "hp:fmt": HELP_FORMATS,
    "hp:faq": HELP_FAQ,
}


async def help__show_section(update: Update, context: ContextTypes.DEFAULT_TYPE,
                             data: str) -> None:
    await update.callback_query.answer()
    text = _HELP_SECTIONS.get(data, HELP_MENU_TEXT)
    await update.callback_query.edit_message_text(
        text=text, reply_markup=kb.help_back(), parse_mode="Markdown",
    )


# ===========================================================================
# Handlers — inspector
# ===========================================================================

_USERNAME_RE = re.compile(r"@([A-Za-z0-9_]{5,})")
_LINK_RE = re.compile(
    r"https?://t(?:elegram)?\.me/([A-Za-z0-9_]{5,})(?:/\d+)?", re.IGNORECASE
)


async def inspector__enter_inspect_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await states.set_state(update.effective_user.id, states.AWAIT_INSPECT)
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.INSPECTOR_PROMPT,
        reply_markup=kb.cancel_back("ins"),
        parse_mode="Markdown",
    )


async def inspector__show_recent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    rows = await repo.recent_inspected(update.effective_user.id, limit=10)
    if not rows:
        text = "🕘 *Recent Inspections*\n\nNo inspections yet."
    else:
        lines = ["🕘 *Recent Inspections*", ""]
        for r in rows:
            date = (r.get("created_at") or "")[:16].replace("T", " ")
            name = r.get("title") or r.get("first_name") or "—"
            uname = f"@{r['username']}" if r.get("username") else "—"
            kind = r.get("chat_type") or "—"
            members = r.get("members")
            ms = f"{members:,}" if members else "—"
            lines.append(f"• `{date}` {name}\n   {uname} · {kind} · 👥 {ms}")
        text = "\n".join(lines)
    await update.callback_query.edit_message_text(
        text=text, reply_markup=kb.inspector_back(), parse_mode="Markdown",
    )


async def inspector__handle_inspect_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    raw = (update.message.text or update.message.caption or "").strip()
    user_id = update.effective_user.id
    if not raw:
        await safe_reply_text(update.message, msg.INSPECTOR_EMPTY)
        return

    username = inspector__extract_username(raw)
    if not username:
        await safe_reply_text(update.message, msg.INSPECTOR_EMPTY)
        return

    status = await safe_reply_text(
        update.message, f"\U0001f50d Looking up @{md_escape(username)}\u2026"
    )
    if status is None:
        return

    # Try getChat with smart retries (no retry on permanent errors).
    chat, error_reason = await _inspector_get_chat_smart(context, f"@{username}")
    if chat is None:
        await safe_edit_message_text(
            context.bot, status.chat_id, status.message_id,
            msg.inspect_failed_text(md_escape(error_reason)),
            reply_markup=kb.inspector_back(),
        )
        return

    info = inspector__chat_to_info(chat)
    # Escape all dynamic string values before rendering as Markdown.
    info = _inspector_escape_info(info)
    await repo.add_inspected_chat(
        user_id,
        chat_id=info.get("chat_id"),
        username=info.get("username"),
        title=info.get("title"),
        chat_type=info.get("type"),
        members=info.get("members"),
        description=info.get("description"),
        first_name=info.get("first_name"),
        last_name=info.get("last_name"),
        bio=info.get("bio"),
        is_bot=info.get("is_bot"),
    )
    await safe_edit_message_text(
        context.bot, status.chat_id, status.message_id,
        msg.inspect_result_text(info),
        reply_markup=kb.inspector_back(),
    )


async def _inspector_get_chat_smart(context, chat_ref) -> tuple[object | None, str]:
    """Resolve a chat via getChat with smart retries.

    Retries ONLY on transient errors (RetryAfter, NetworkError, TimedOut).
    Permanent errors (BadRequest \"not found\", Forbidden) fail immediately.
    Returns ``(chat, error_reason)`` — on success error_reason is \"\".
    """
    max_retries = 3
    base_delay = 2.0
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            chat = await context.bot.get_chat(chat_ref)
            return chat, ""
        except RetryAfter as exc:
            wait = min(exc.retry_after + 1, 30)
            logger.info("Rate limited on getChat: %ss (attempt %d/%d)",
                        exc.retry_after, attempt, max_retries)
            await asyncio.sleep(wait)
            last_exc = exc
            continue
        except BadRequest as exc:
            # Permanent — don't retry.
            msg_text = str(exc).lower()
            logger.info("get_chat failed for %s: %s", chat_ref, exc)
            if "chat not found" in msg_text:
                return None, (
                    f"Chat {chat_ref} was not found.\n\n"
                    "Possible reasons:\n"
                    "\u2022 The username doesn't exist or was changed\n"
                    "\u2022 The chat is private and not accessible to bots\n"
                    "\u2022 The username belongs to a deleted account\n"
                    "\u2022 You typed the username incorrectly"
                )
            return None, f"Telegram rejected the lookup: {exc}"
        except Forbidden as exc:
            logger.info("get_chat forbidden for %s: %s", chat_ref, exc)
            return None, (
                f"The bot doesn't have permission to access {chat_ref}. "
                "Private chats require the bot to be a member."
            )
        except (NetworkError, TimedOut) as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = min(base_delay * (2 ** (attempt - 1)), 30.0)
                logger.info("Network error on getChat (attempt %d/%d), "
                            "retrying in %.1fs: %s", attempt, max_retries, delay, exc)
                await asyncio.sleep(delay)
                continue
            return None, f"Network error after {max_retries} retries: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_chat unexpected error for %s: %s", chat_ref, exc)
            return None, f"Unexpected error: {exc}"

    return None, f"Request failed after {max_retries} retries: {last_exc}"


def inspector__extract_username(text: str) -> str | None:
    m = _USERNAME_RE.search(text)
    if m:
        return m.group(1)
    m = _LINK_RE.search(text)
    if m:
        return m.group(1)
    bare = text.strip().lstrip("@")
    if re.fullmatch(r"[A-Za-z0-9_]{5,}", bare):
        return bare
    return None


def inspector__chat_to_info(chat) -> dict[str, Any]:
    info: dict[str, Any] = {}
    info["chat_id"] = getattr(chat, "id", None)
    info["type"] = inspector__chat_type(chat)
    info["username"] = getattr(chat, "username", None)
    info["title"] = getattr(chat, "title", None)
    info["first_name"] = getattr(chat, "first_name", None)
    info["last_name"] = getattr(chat, "last_name", None)
    info["description"] = getattr(chat, "description", None)
    info["bio"] = getattr(chat, "bio", None)
    info["is_bot"] = getattr(chat, "is_bot", None)
    info["members"] = getattr(chat, "member_count", None)
    return info


def inspector__chat_type(chat) -> str:
    t = getattr(chat, "type", None)
    mapping = {
        "private": "Private user",
        "group": "Group",
        "supergroup": "Supergroup",
        "channel": "Channel",
    }
    return mapping.get(t, t or "Chat")


def _inspector_escape_info(info: dict) -> dict:
    """Return a copy of *info* with all string values Markdown-escaped."""
    out = {}
    for k, v in info.items():
        if isinstance(v, str):
            out[k] = md_escape(v)
        else:
            out[k] = v
    return out


# ===========================================================================
# Handlers — toolbox
# ===========================================================================

_TOOL_MEDIA = {
    "audio": {"video", "gif", "video_note"},
    "thumb": {"video", "gif", "video_note", "photo", "document"},
    "info": {"video", "audio", "gif", "video_note", "photo",
             "document", "voice", "sticker"},
    "compress": {"video", "gif", "video_note"},
    "imgconv": {"photo", "document"},
}


async def toolbox__handle_tool_selection(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                         tool: str) -> None:
    """Called from the menu router when a tb:<tool> button is pressed."""
    if tool not in _TOOL_MEDIA:
        await update.callback_query.answer("Unknown tool.")
        return
    await states.set_state(update.effective_user.id, states.AWAIT_TOOLBOX)
    await states.set_tool(update.effective_user.id, tool)
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.TOOLBOX_PROMPTS[tool],
        reply_markup=kb.cancel_back("tb"),
        parse_mode="Markdown",
    )


async def toolbox__handle_forwarded_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    message = update.message
    tool = await states.get_tool(user_id)
    if not tool:
        await message.reply_text(
            "⚠️ No tool selected. Pick one from the Media Toolbox menu.",
            reply_markup=kb.toolbox_menu(),
        )
        return

    media_type = telegram_media_type(message)
    if media_type is None or media_type not in _TOOL_MEDIA.get(tool, set()):
        await message.reply_text(msg.TOOLBOX_INVALID_MEDIA, parse_mode="Markdown")
        return

    if tool == "imgconv":
        await states.set_state(user_id, states.AWAIT_TOOLBOX)
        await states.set_tool(user_id, "imgconv:msg:" + str(message.message_id))
        await message.reply_text(
            "🔄 Send the target format as a message "
            "(`png`, `jpg`, `webp`, or `bmp`):",
            parse_mode="Markdown",
            reply_markup=kb.cancel_back("tb"),
        )
        context.user_data["imgconv_msg_id"] = message.message_id
        context.user_data["imgconv_chat_id"] = message.chat_id
        return

    status = await message.reply_text(
        f"🧰 *{tool.title()}* — downloading your media…",
        parse_mode="Markdown",
        reply_markup=kb.cancel_back("tb"),
    )
    asyncio.create_task(
        toolbox__run_tool(user_id, tool, message, context,
                          status.chat_id, status.message_id, media_type)
    )


async def toolbox__handle_imgconv_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Second step of image conversion: user sent the target format."""
    user_id = update.effective_user.id
    fmt = (update.message.text or "").strip().lower().lstrip(".")
    if fmt not in ("png", "jpg", "jpeg", "webp", "bmp"):
        await update.message.reply_text(
            "⚠️ Unsupported format. Send one of: `png`, `jpg`, `webp`, `bmp`.",
            parse_mode="Markdown",
        )
        return

    src_msg_id = context.user_data.get("imgconv_msg_id")
    src_chat_id = context.user_data.get("imgconv_chat_id")
    if not src_msg_id or not src_chat_id:
        await update.message.reply_text(
            "⚠️ Session expired. Please restart image conversion from the menu.",
            reply_markup=kb.toolbox_menu(),
        )
        await states.reset(user_id)
        return

    try:
        src_message = await context.bot.forward_message(
            chat_id=update.effective_chat.id,
            from_chat_id=src_chat_id,
            message_id=src_msg_id,
        )
    except Exception as exc:  # noqa: BLE001
        await update.message.reply_text(f"⚠️ Could not retrieve source media: {exc}")
        return

    status = await update.message.reply_text(
        "🔄 Converting image…",
        reply_markup=kb.cancel_back("tb"),
    )
    await states.set_tool(user_id, "imgconv")
    asyncio.create_task(
        toolbox__run_tool(user_id, "imgconv", src_message, context,
                          status.chat_id, status.message_id, "photo",
                          extra={"fmt": fmt})
    )


async def toolbox__run_tool(
    user_id: int, tool: str, message, context,
    chat_id: int, status_msg_id: int, media_type: str,
    *, extra: dict | None = None,
) -> None:
    src_path: Path | None = None
    out_path: Path | None = None
    try:
        await toolbox__edit(context, chat_id, status_msg_id, "📥 Downloading source media…")
        async with download_slot():
            src_path, name, mtype, size = await downloader.download_forwarded(
                message, progress=None, bot=context.bot
            )

        await toolbox__edit(context, chat_id, status_msg_id, f"🧰 Running {tool}…")
        out_path = await toolbox__dispatch_tool(tool, src_path, extra)

        await toolbox__edit(context, chat_id, status_msg_id, "📤 Sending result…")
        ok = await toolbox__send_result(context, chat_id, tool, out_path, src_path, name)
        if not ok:
            await toolbox__edit(context, chat_id, status_msg_id,
                                "⚠️ Could not send the result (size limit).",
                                reply_markup=kb.toolbox_menu())

        await repo.create_task(user_id, f"toolbox:{tool}", "done")
        await states.reset(user_id)

    except media_tools.ToolError as exc:
        logger.warning("tool %s failed: %s", tool, exc)
        await toolbox__edit(context, chat_id, status_msg_id,
                            f"❌ *Tool failed:* `{exc}`",
                            reply_markup=kb.toolbox_menu())
        await repo.create_task(user_id, f"toolbox:{tool}", "failed")
    except downloader.FileTooLarge as exc:
        await toolbox__edit(context, chat_id, status_msg_id,
                            msg.file_too_big_text(config.max_file_size_mb),
                            reply_markup=kb.toolbox_menu())
    except Exception as exc:  # noqa: BLE001
        logger.exception("toolbox %s error: %s", tool, exc)
        await toolbox__edit(context, chat_id, status_msg_id,
                            f"❌ *Error:* `{exc}`",
                            reply_markup=kb.toolbox_menu())
    finally:
        if out_path:
            await remove_path(out_path)
        if src_path:
            await remove_path(src_path)


async def toolbox__dispatch_tool(tool: str, src: Path, extra: dict | None) -> Path:
    if tool == "audio":
        return await media_tools.extract_audio(src, config.frames_dir)
    if tool == "thumb":
        return await media_tools.extract_thumbnail(src, config.frames_dir)
    if tool == "info":
        return src  # placeholder; caller checks tool == "info"
    if tool == "compress":
        return await media_tools.compress_video(src, config.frames_dir)
    if tool == "imgconv":
        fmt = (extra or {}).get("fmt", "png")
        return await media_tools.convert_image(src, config.frames_dir, fmt)
    raise media_tools.ToolError(f"Unknown tool: {tool}")


async def toolbox__send_result(context, chat_id: int, tool: str, out_path: Path,
                               src_path: Path, src_name: str) -> bool:
    if tool == "info":
        info = await media_tools.media_info(src_path)
        await context.bot.send_message(
            chat_id=chat_id,
            text=msg.media_info_text(info),
            parse_mode="Markdown",
            reply_markup=kb.toolbox_back(),
        )
        return True

    caption = f"🧰 {tool.title()} result for `{src_name}`"
    result = await send_file_back(
        chat_id, out_path, caption, context, progress_cb=None
    )
    if result == "ok":
        try:
            await context.bot.send_message(
                chat_id=chat_id, text="✅ Done.",
                reply_markup=kb.toolbox_back(),
            )
        except Exception:  # noqa: BLE001
            pass
        return True
    return False


async def toolbox__edit(context, chat_id: int, message_id: int, text: str,
                        reply_markup=None) -> None:
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=message_id,
            text=text, parse_mode="Markdown", reply_markup=reply_markup,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("toolbox edit skipped: %s", exc)


# ===========================================================================
# Handlers — library
# ===========================================================================


async def library__show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.LIBRARY_MENU_TEXT,
        reply_markup=kb.library_menu(),
        parse_mode="Markdown",
    )


async def library__handle_action(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                 data: str) -> None:
    """Dispatch lib:<action> callbacks."""
    q = update.callback_query
    user_id = update.effective_user.id
    action = data.split(":", 1)[1]

    if action == "browse":
        await library__browse(update, context, None)
    elif action == "search":
        await states.set_state(user_id, states.AWAIT_LIBRARY_SEARCH)
        await q.answer()
        await q.edit_message_text(
            text="🔍 *Library search*\n\nSend me a keyword to search your "
                 "library (filename, note or tags).",
            reply_markup=kb.cancel_back("lib"),
            parse_mode="Markdown",
        )
    elif action == "clear":
        n = await repo.library_clear(user_id)
        await q.answer(f"Cleared {n} entries.")
        await q.edit_message_text(
            text=f"🧹 *Library cleared* ({n} entries removed).",
            reply_markup=kb.library_menu(),
            parse_mode="Markdown",
        )
    elif action.startswith("t:"):
        media_type = action.split(":", 1)[1]
        await library__browse(update, context, media_type)
    elif action.startswith("del:"):
        entry_id = int(action.split(":", 1)[1])
        ok = await repo.library_remove(user_id, entry_id)
        await q.answer("Removed." if ok else "Not found.")
        await library__browse(update, context, None)
    elif action == "savelast":
        await library__save_last(update, context)
    else:
        await q.answer("Unknown action.")


async def library__browse(update: Update, context: ContextTypes.DEFAULT_TYPE,
                          media_type: str | None) -> None:
    user_id = update.effective_user.id
    rows = await repo.library_entries(user_id, limit=15, media_type=media_type)
    title = "📚 Library" + (f" · {media_type}" if media_type else "")
    text = msg.library_list_text(rows, title)
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=text, reply_markup=kb.library_back(), parse_mode="Markdown",
    )


async def library__handle_search_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    query = (update.message.text or "").strip()
    if not query:
        await update.message.reply_text("⚠️ Please send a search keyword.")
        return
    rows = await repo.library_search(user_id, query, limit=20)
    text = msg.library_list_text(rows, f"🔍 Search: {query}")
    await update.message.reply_text(
        text=text, reply_markup=kb.library_back(), parse_mode="Markdown",
    )


async def library__save_last(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    meta = await states.get_last_download(user_id)
    if not meta:
        await update.callback_query.answer(
            "No recent download to save.", show_alert=True
        )
        return
    entry_id = await repo.add_library_entry(
        user_id,
        file_name=meta.get("file_name", "media"),
        media_type=meta.get("media_type"),
        file_size=meta.get("file_size", 0),
        file_id=meta.get("file_id"),
    )
    await update.callback_query.answer(f"Saved as #{entry_id} ⭐")
    try:
        await update.callback_query.edit_message_text(
            text=(
                f"⭐ *Saved to Library* (#{entry_id})\n\n"
                f"📄 `{meta.get('file_name','?')}`\n"
                f"🗂 {meta.get('media_type','—')} · "
                f"{human_size(meta.get('file_size',0))}"
            ),
            reply_markup=kb.library_back(),
            parse_mode="Markdown",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("library save edit skipped: %s", exc)


# ===========================================================================
# Handlers — stats
# ===========================================================================


async def stats__show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.STATS_MENU_TEXT,
        reply_markup=kb.stats_menu(),
        parse_mode="Markdown",
    )


async def stats__show_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    s = await repo.user_stats(update.effective_user.id)
    try:
        await update.callback_query.edit_message_text(
            text=msg.user_stats_text(s),
            reply_markup=kb.stats_back(),
            parse_mode="Markdown",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("stats user edit skipped: %s", exc)


async def stats__show_global(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    s = await repo.global_stats()
    try:
        await update.callback_query.edit_message_text(
            text=msg.global_stats_text(s),
            reply_markup=kb.stats_back(),
            parse_mode="Markdown",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("stats global edit skipped: %s", exc)


# ===========================================================================
# Handlers — qr
# ===========================================================================


async def qr__show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.QR_MENU_TEXT,
        reply_markup=kb.qr_menu(),
        parse_mode="Markdown",
    )


async def qr__enter_make_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await states.set_state(update.effective_user.id, states.AWAIT_QR)
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.QR_PROMPT,
        reply_markup=kb.cancel_back("qr"),
        parse_mode="Markdown",
    )


async def qr__handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or update.message.caption or "").strip()
    if not text:
        await update.message.reply_text("⚠️ Please send some text or a link.")
        return

    status = await update.message.reply_text("🔳 Generating QR code…")
    out_path: Path | None = None
    try:
        out_path = await qr_generator.generate_qr(text, config.frames_dir)
        with out_path.open("rb") as fh:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=fh,
                caption=f"🔳 QR code for:\n`{text[:200]}`",
                parse_mode="Markdown",
                reply_markup=kb.qr_back(),
            )
        await states.reset(update.effective_user.id)
    except qr_generator.QRError as exc:
        await qr__edit(context, status.chat_id, status.message_id,
                       f"❌ *QR error:* `{exc}`",
                       reply_markup=kb.qr_back())
    except Exception as exc:  # noqa: BLE001
        logger.exception("qr generation failed: %s", exc)
        await qr__edit(context, status.chat_id, status.message_id,
                       f"❌ *Error:* `{exc}`",
                       reply_markup=kb.qr_back())
    finally:
        if out_path:
            await remove_path(out_path)


async def qr__edit(context, chat_id: int, message_id: int, text: str,
                   reply_markup=None) -> None:
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=message_id,
            text=text, parse_mode="Markdown", reply_markup=reply_markup,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("qr edit skipped: %s", exc)


# ===========================================================================
# Handlers — batch
# ===========================================================================

_BATCH_WINDOW_SECONDS = 6.0
_batch_collectors: dict[int, "_AlbumCollector"] = {}


class _AlbumCollector:
    """Accumulates messages belonging to one media group for a user."""

    def __init__(self, user_id: int, chat_id: int, status_msg_id: int, bot):
        self.user_id = user_id
        self.chat_id = chat_id
        self.status_msg_id = status_msg_id
        self.bot = bot
        self.messages: list = []
        self.timer: asyncio.Task | None = None
        self.lock = asyncio.Lock()

    async def add(self, message) -> None:
        async with self.lock:
            self.messages.append(message)
            count = len(self.messages)
        if self.timer and not self.timer.done():
            self.timer.cancel()
        self.timer = asyncio.create_task(self._close_after_window(count))

    async def _close_after_window(self, last_count: int) -> None:
        try:
            await asyncio.sleep(_BATCH_WINDOW_SECONDS)
        except asyncio.CancelledError:
            return
        async with self.lock:
            if len(self.messages) != last_count:
                return  # more arrived; a newer timer will handle it
            msgs = list(self.messages)
            self.messages.clear()
        await batch__process_album(self, msgs)


async def batch__enter_batch_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await states.set_state(update.effective_user.id, states.AWAIT_BATCH)
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.BATCH_PROMPT,
        reply_markup=kb.cancel_back("batch"),
        parse_mode="Markdown",
    )


async def batch__handle_batch_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    user_id = update.effective_user.id
    media_type = telegram_media_type(message)
    if media_type is None:
        await message.reply_text(
            "⚠️ That message has no media. Forward an album or several media messages."
        )
        return

    collector = _batch_collectors.get(user_id)
    if collector is None:
        status = await message.reply_text(
            "📦 Collecting album… send more items, or wait ~6s to finish.",
            reply_markup=kb.cancel_back("batch"),
        )
        collector = _AlbumCollector(user_id, status.chat_id, status.message_id,
                                    context.bot)
        _batch_collectors[user_id] = collector
    await collector.add(message)


async def batch__process_album(collector: _AlbumCollector, messages: list) -> None:
    user_id = collector.user_id
    _batch_collectors.pop(user_id, None)
    await states.reset(user_id)

    bot = collector.bot

    paths: list[Path] = []
    total_size = 0
    try:
        await batch__edit(bot, collector.chat_id, collector.status_msg_id,
                          f"📥 Downloading {len(messages)} items…")
        for i, m in enumerate(messages, 1):
            await batch__edit(bot, collector.chat_id, collector.status_msg_id,
                              f"📥 Downloading item {i}/{len(messages)}…")
            async with download_slot():
                path, name, mtype, size = await downloader.download_forwarded(
                    m, progress=None, bot=bot
                )
            paths.append(path)
            total_size += size
            await repo.add_download_history(
                user_id, file_name=name, file_size=size, mime_type=None,
                media_type=mtype, source="batch", status="done",
                file_unique_id=getattr(m.document, "file_unique_id", None)
                if m.document else None,
            )

        await batch__edit(bot, collector.chat_id, collector.status_msg_id,
                          msg.batch_summary_text(len(paths), total_size))

        await batch__send_back(bot, collector.chat_id, paths)

        settings = await repo.ensure_settings(user_id)
        if settings.get("auto_delete"):
            for p in paths:
                await remove_path(p)

    except downloader.FileTooLarge as exc:
        await batch__edit(bot, collector.chat_id, collector.status_msg_id,
                          msg.file_too_big_text(config.max_file_size_mb),
                          reply_markup=kb.batch_back())
    except Exception as exc:  # noqa: BLE001
        logger.exception("album processing failed: %s", exc)
        await batch__edit(bot, collector.chat_id, collector.status_msg_id,
                          msg.batch_failed_text(str(exc)),
                          reply_markup=kb.batch_back())
    finally:
        for p in paths:
            try:
                if p.exists():
                    await remove_path(p)
            except Exception:  # noqa: BLE001
                pass


async def batch__send_back(bot, chat_id: int, paths: list[Path]) -> None:
    """Send the downloaded files back — as an album when possible."""
    if not paths:
        return
    if len(paths) == 1:
        p = paths[0]
        if p.stat().st_size <= config.upload_limit_bytes:
            try:
                async with p.open("rb") as fh:
                    await bot.send_document(chat_id=chat_id, document=fh,
                                            filename=p.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("album single send failed: %s", exc)
        return
    for p in paths:
        if p.stat().st_size > config.upload_limit_bytes:
            await bot.send_message(
                chat_id=chat_id,
                text=f"ℹ️ `{p.name}` is too large to send back via the Bot API.",
                parse_mode="Markdown",
            )
            continue
        try:
            async with p.open("rb") as fh:
                await bot.send_document(chat_id=chat_id, document=fh,
                                        filename=p.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("album item send failed: %s", exc)


async def batch__edit(bot, chat_id: int, message_id: int, text: str,
                      reply_markup=None) -> None:
    try:
        await bot.edit_message_text(
            chat_id=chat_id, message_id=message_id,
            text=text, parse_mode="Markdown", reply_markup=reply_markup,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("album edit skipped: %s", exc)


# ===========================================================================
# Handlers — scheduled
# ===========================================================================


async def scheduled__show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.SCHEDULED_MENU_TEXT,
        reply_markup=kb.scheduled_menu(),
        parse_mode="Markdown",
    )


async def scheduled__show_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    rows = await repo.list_scheduled_tasks(update.effective_user.id, limit=15)
    await update.callback_query.edit_message_text(
        text=msg.scheduled_list_text(rows),
        reply_markup=kb.scheduled_back(),
        parse_mode="Markdown",
    )


async def scheduled__handle_schedule_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """User sent a time string; validate and ask for media."""
    text = (update.message.text or "").strip()
    run_at = scheduled__parse_time(text)
    if not run_at:
        await update.message.reply_text(
            msg.SCHEDULE_INVALID, parse_mode="Markdown",
            reply_markup=kb.cancel_back("sched"),
        )
        return
    context.user_data["sched_run_at"] = run_at
    await states.set_state(update.effective_user.id, states.AWAIT_DOWNLOAD_FORWARD)
    context.user_data["sched_mode"] = True
    await update.message.reply_text(
        f"⏰ Scheduled for *{run_at}*. Now forward the media to queue.",
        parse_mode="Markdown",
        reply_markup=kb.cancel_back("sched"),
    )


def scheduled__parse_time(text: str) -> str | None:
    """Parse 'YYYY-MM-DD HH:MM' into an ISO datetime string, or None."""
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return None


async def scheduled__queue_scheduled(
    user_id: int, run_at: str, kind: str, payload: dict,
) -> int:
    """Persist a scheduled task."""
    return await repo.add_scheduled_task(
        user_id, kind, json.dumps(payload, default=str), run_at
    )


async def scheduled__run_due_tasks(app) -> int:
    """Background loop tick: run any pending scheduled downloads that are due."""
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M")
    due = await repo.pending_scheduled_tasks(now_iso, limit=20)
    for task in due:
        await repo.update_scheduled_task(task["id"], "done")
        try:
            await app.bot.send_message(
                chat_id=task["user_id"],
                text=(
                    f"⏰ *Scheduled task #{task['id']} was due.*\n\n"
                    f"Kind: `{task['kind']}`\n"
                    f"Scheduled for: `{task['run_at']}`\n\n"
                    "_Note: deferred media redownload isn't possible via the "
                    "Bot API without the original message — re-forward the "
                    "media to run it now._"
                ),
                parse_mode="Markdown",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("scheduled notify failed: %s", exc)
    return len(due)


# ===========================================================================
# Handlers — backup_restore
# ===========================================================================


async def backup__show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.BACKUP_MENU_TEXT,
        reply_markup=kb.backup_menu(),
        parse_mode="Markdown",
    )


async def backup__do_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    await update.callback_query.answer("Exporting…")
    await update.callback_query.edit_message_text(
        text=msg.BACKUP_EXPORTING, reply_markup=kb.backup_back(),
        parse_mode="Markdown",
    )
    try:
        data = await repo.export_user_data(user_id)
        path = await backup.export_user_json(user_id)
        with path.open("rb") as fh:
            await context.bot.send_document(
                chat_id=user_id, document=fh, filename=path.name,
                caption=msg.backup_export_done_text(
                    path.name, path.stat().st_size,
                    {k: len(v) if isinstance(v, list) else 1
                     for k, v in data.items()
                     if k != "exported_at"},
                ),
                parse_mode="Markdown",
            )
        await remove_path(path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("backup export failed: %s", exc)
        await context.bot.send_message(
            chat_id=user_id, text=msg.backup_failed_text(str(exc)),
            parse_mode="Markdown", reply_markup=kb.backup_back(),
        )


async def backup__enter_import(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await states.set_state(update.effective_user.id, states.AWAIT_BACKUP_IMPORT)
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.BACKUP_IMPORT_PROMPT,
        reply_markup=kb.cancel_back("bk"),
        parse_mode="Markdown",
    )


async def backup__handle_import_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    doc = update.message.document
    if doc is None:
        await update.message.reply_text("⚠️ Please send a JSON backup file.")
        return

    try:
        path, _, _, _ = await downloader.download_forwarded(
            update.message, progress=None, bot=context.bot
        )
    except Exception as exc:  # noqa: BLE001
        await update.message.reply_text(f"⚠️ Could not download the file: `{exc}`",
                                        parse_mode="Markdown")
        return

    try:
        data = await backup.parse_backup_file(path)
        summary = await backup.restore_from_dict(user_id, data)
        await update.message.reply_text(
            msg.backup_import_done_text(summary),
            reply_markup=kb.backup_back(),
            parse_mode="Markdown",
        )
    except json.JSONDecodeError as exc:
        await update.message.reply_text(
            f"⚠️ Invalid JSON file: `{exc}`", parse_mode="Markdown")
    except Exception as exc:  # noqa: BLE001
        logger.exception("backup import failed: %s", exc)
        await update.message.reply_text(
            msg.backup_failed_text(str(exc)),
            parse_mode="Markdown", reply_markup=kb.backup_back(),
        )
    finally:
        await remove_path(path)
        await states.reset(user_id)


# ===========================================================================
# Handlers — admin
# ===========================================================================


def admin__is_admin(user_id: int) -> bool:
    return user_id in config.admin_ids


async def admin__show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not admin__is_admin(update.effective_user.id):
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(
            text=msg.ADMIN_DENIED,
            reply_markup=kb.back_only("main"),
            parse_mode="Markdown",
        )
        return
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.ADMIN_MENU_TEXT,
        reply_markup=kb.admin_menu(),
        parse_mode="Markdown",
    )


async def admin__list_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not admin__is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer()
    users = await repo.all_users(limit=20)
    total = await repo.user_count()
    await update.callback_query.edit_message_text(
        text=msg.admin_users_text(users, total),
        reply_markup=kb.admin_back(),
        parse_mode="Markdown",
    )


async def admin__global_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not admin__is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer()
    s = await repo.global_stats()
    await update.callback_query.edit_message_text(
        text=msg.global_stats_text(s),
        reply_markup=kb.admin_back(),
        parse_mode="Markdown",
    )


async def admin__export_global(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not admin__is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer("Exporting…")
    try:
        path = await backup.export_global_json()
        with path.open("rb") as fh:
            await context.bot.send_document(
                chat_id=update.effective_user.id, document=fh,
                filename=path.name,
                caption="💾 Global stats export",
            )
        await remove_path(path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("global export failed: %s", exc)
        await update.callback_query.answer(f"Failed: {exc}")


async def admin__enter_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not admin__is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await states.set_state(update.effective_user.id, states.AWAIT_ADMIN_BCAST)
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.ADMIN_BCAST_PROMPT,
        reply_markup=kb.cancel_back("admin"),
        parse_mode="Markdown",
    )


async def admin__handle_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    admin_id = update.effective_user.id
    if not admin__is_admin(admin_id):
        await update.message.reply_text("🚫 Not authorised.")
        return
    text = (update.message.text or "").strip()
    if not text:
        await update.message.reply_text("⚠️ Send the broadcast text.")
        return
    await states.reset(admin_id)

    status = await update.message.reply_text("📣 Broadcasting…")
    users = await repo.all_users(limit=10000)
    bid = await repo.create_broadcast(admin_id, text)
    sent = failed = 0
    for u in users:
        try:
            await context.bot.send_message(
                chat_id=u["tg_id"],
                text=f"📣 *Broadcast:*\n\n{text}",
                parse_mode="Markdown",
            )
            sent += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("broadcast to %s failed: %s", u["tg_id"], exc)
            failed += 1
    await repo.update_broadcast_counts(bid, sent, failed)
    try:
        await context.bot.edit_message_text(
            chat_id=status.chat_id, message_id=status.message_id,
            text=msg.admin_bcast_done_text(sent, failed),
            parse_mode="Markdown", reply_markup=kb.admin_back(),
        )
    except Exception:  # noqa: BLE001
        pass


# ===========================================================================
# Handlers — mtproto_admin (Task ID 8)
# ===========================================================================
# Admin-only controls for the MTProto (Telethon) backend:
#   * Status view (uptime + stats)
#   * Start / stop / restart
#   * Screenshot a chat by @username / id
#   * Download restricted content via MTProto
#
# All Telethon access goes through the ``mtproto_manager`` / ``mtproto_service``
# SimpleNamespaces defined above. Telethon itself is lazy-imported inside those
# modules, so these handlers are safe to load even when Telethon is absent.


def mtproto__is_admin(user_id: int) -> bool:
    return user_id in config.admin_ids


def _mtproto_status_text(status: dict) -> str:
    s = status
    uptime = s.get("uptime_seconds", 0)
    h, rem = divmod(int(uptime), 3600)
    m, sec = divmod(rem, 60)
    stats = s.get("stats", {})
    sess = s.get("session", {})
    started = "✅ Running" if s.get("started") else "⚪️ Stopped"
    return (
        f"*🛰️ MTProto Backend*\n\n"
        f"Status: {started}\n"
        f"Uptime: `{h}:{m:02d}:{sec:02d}`\n"
        f"Telethon installed: {'✅' if s.get('available') else '❌'}\n"
        f"Configured: {'✅' if s.get('configured') else '❌'}\n\n"
        f"*Session:*\n"
        f"  Name: `{md_escape(sess.get('session_name','—'))}`\n"
        f"  File exists: {'✅' if sess.get('session_exists') else '❌'}\n"
        f"  API ID set: {'✅' if sess.get('api_id_set') else '❌'}\n\n"
        f"*Stats:*\n"
        f"  Requests: {stats.get('requests',0)}\n"
        f"  Successes: {stats.get('successes',0)}\n"
        f"  Failures: {stats.get('failures',0)}\n"
        f"  FloodWaits: {stats.get('flood_waits',0)}\n"
        f"  Reconnects: {stats.get('reconnects',0)}\n"
    )


async def mtproto__show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not mtproto__is_admin(update.effective_user.id):
        await update.callback_query.answer()
        await safe_edit_message_text(
            context.bot, update.effective_chat.id,
            update.callback_query.message.message_id,
            msg.ADMIN_DENIED, reply_markup=kb.back_only("main"),
        )
        return
    await update.callback_query.answer()
    status = mtproto_manager.get_status()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        _mtproto_status_text(status),
        reply_markup=kb.mtproto_menu(),
    )


async def mtproto__start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not mtproto__is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer("Starting MTProto…")
    result = await mtproto_manager.start()
    status = mtproto_manager.get_status()
    text = (f"*🛰️ MTProto Start*\n\n"
            f"Result: `{md_escape(result)}`\n\n" + _mtproto_status_text(status))
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        text, reply_markup=kb.mtproto_menu(),
    )


async def mtproto__stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not mtproto__is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer("Stopping MTProto…")
    result = await mtproto_manager.stop()
    status = mtproto_manager.get_status()
    text = (f"*🛰️ MTProto Stop*\n\n"
            f"Result: `{md_escape(result)}`\n\n" + _mtproto_status_text(status))
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        text, reply_markup=kb.mtproto_menu(),
    )


async def mtproto__restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not mtproto__is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer("Restarting MTProto…")
    result = await mtproto_manager.restart()
    status = mtproto_manager.get_status()
    text = (f"*🛰️ MTProto Restart*\n\n"
            f"Result: `{md_escape(result)}`\n\n" + _mtproto_status_text(status))
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        text, reply_markup=kb.mtproto_menu(),
    )


async def mtproto__refresh_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not mtproto__is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer()
    status = mtproto_manager.get_status()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        _mtproto_status_text(status), reply_markup=kb.mtproto_menu(),
    )


async def mtproto__enter_screenshot_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not mtproto__is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await states.set_state(update.effective_user.id, states.AWAIT_MTPROTO_SCREENSHOT)
    await update.callback_query.answer()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        "📸 *Screenshot mode*\n\n"
        "Send me a @username or chat id to capture a text-rendered screenshot "
        "of its recent messages (via MTProto).",
        reply_markup=kb.cancel_back("mtproto"),
    )


async def mtproto__handle_screenshot_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not mtproto__is_admin(user_id):
        await safe_reply_text(update.message, "🚫 Not authorised.")
        return
    chat_ref = (update.message.text or "").strip()
    if not chat_ref:
        await safe_reply_text(update.message, "⚠️ Send a @username or chat id.")
        return
    await states.reset(user_id)

    status_msg = await safe_reply_text(update.message, "📸 Capturing screenshot…")
    if status_msg is None:
        return

    try:
        path = await mtproto_service.screenshot(chat_ref)
        if path and path.exists():
            with path.open("rb") as fh:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=fh,
                    caption=f"📸 Screenshot of `{md_escape(chat_ref)}`",
                    parse_mode="Markdown",
                )
            await remove_path(path)
            await safe_edit_message_text(
                context.bot, status_msg.chat_id, status_msg.message_id,
                "✅ Screenshot sent.", reply_markup=kb.mtproto_menu(),
            )
        else:
            await safe_edit_message_text(
                context.bot, status_msg.chat_id, status_msg.message_id,
                "❌ Screenshot failed. Is MTProto running and the chat accessible?",
                reply_markup=kb.mtproto_menu(),
            )
    except mtproto_service.MTProtoError as exc:
        await safe_edit_message_text(
            context.bot, status_msg.chat_id, status_msg.message_id,
            f"❌ `{md_escape(str(exc))}`",
            reply_markup=kb.mtproto_menu(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("screenshot failed: %s", exc)
        await safe_edit_message_text(
            context.bot, status_msg.chat_id, status_msg.message_id,
            f"❌ Error: `{md_escape(str(exc))}`",
            reply_markup=kb.mtproto_menu(),
        )


async def mtproto__enter_download_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not mtproto__is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await states.set_state(update.effective_user.id, states.AWAIT_MTPROTO_DOWNLOAD)
    await update.callback_query.answer()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        "📥 *MTProto download mode*\n\n"
        "Send me any of:\n"
        "• A message link: `https://t.me/channel/123`\n"
        "• `@username message_id` (e.g. `@channel 123`)\n"
        "• Just `@username` — I'll show recent messages so you can pick an id\n"
        "• A private link: `https://t.me/c/1234567890/123`\n\n"
        "_Downloads via MTProto can access restricted content the userbot "
        "can see._",
        reply_markup=kb.cancel_back("mtproto"),
    )


async def mtproto__handle_download_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not mtproto__is_admin(user_id):
        await safe_reply_text(update.message, "🚫 Not authorised.")
        return
    raw = (update.message.text or "").strip()
    if not raw:
        await safe_reply_text(update.message, "⚠️ Send a link or @username.")
        return

    # Case 1: input includes a message id → download directly.
    try:
        chat_ref, msg_id = _mtproto_parse_input(raw)
    except mtproto_service.MTProtoError:
        # Case 2: input is just a @username / link without msg id → show history.
        await _mtproto_show_history_and_ask_msgid(update, context, raw)
        return

    await states.reset(user_id)
    await _mtproto_do_download(update, context, chat_ref, msg_id)


async def mtproto__handle_msgid_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin sent a message id after we showed the channel history."""
    user_id = update.effective_user.id
    if not mtproto__is_admin(user_id):
        await safe_reply_text(update.message, "🚫 Not authorised.")
        return
    text = (update.message.text or "").strip()
    try:
        msg_id = int(text)
    except ValueError:
        await safe_reply_text(
            update.message,
            "⚠️ Send a numeric message id (e.g. `123`).",
            reply_markup=kb.cancel_back("mtproto"),
        )
        return
    if msg_id <= 0:
        await safe_reply_text(update.message, "⚠️ Message id must be positive.")
        return

    chat_ref = context.user_data.get("mtproto_chat_ref")
    if not chat_ref:
        await safe_reply_text(
            update.message,
            "⚠️ Session expired. Send the @username again.",
            reply_markup=kb.mtproto_menu(),
        )
        await states.reset(user_id)
        return

    await states.reset(user_id)
    await _mtproto_do_download(update, context, chat_ref, msg_id)


async def _mtproto_show_history_and_ask_msgid(update: Update,
                                              context: ContextTypes.DEFAULT_TYPE,
                                              raw: str) -> None:
    """Resolve a @username/link via MTProto, show recent messages, ask for id."""
    status_msg = await safe_reply_text(
        update.message, "📥 Resolving chat via MTProto…"
    )
    if status_msg is None:
        return

    try:
        # Normalize the input to a chat reference.
        chat_ref = _mtproto_normalize_chat_ref(raw)
        if not chat_ref:
            await safe_edit_message_text(
                context.bot, status_msg.chat_id, status_msg.message_id,
                "❌ Could not parse that. Send a @username or t.me link.",
                reply_markup=kb.mtproto_menu(),
            )
            return

        # Resolve + fetch history via MTProto.
        info = await mtproto_service.resolve_entity(chat_ref)
        history = await mtproto_service.get_channel_history(chat_ref, limit=15)

        # Stash for the message-id step.
        context.user_data["mtproto_chat_ref"] = chat_ref
        await states.set_state(update.effective_user.id, states.AWAIT_MTPROTO_MSGID)

        title = md_escape(info.get("title") or info.get("first_name")
                          or str(info.get("id", chat_ref)))
        uname = info.get("username")
        uname_s = f"@{md_escape(uname)}" if uname else "—"
        members = info.get("participants_count")
        members_s = f"{members:,}" if members else "—"

        lines = [
            "✅ *Chat resolved via MTProto*\n",
            f"📛 *Name:* {title}",
            f"🔗 *Username:* {uname_s}",
            f"👥 *Members:* {members_s}",
            f"🆔 *ID:* `{info.get('id', '—')}`",
            "",
            "*Recent messages (newest first):*",
            "",
        ]
        media_count = 0
        for m in history:
            mid = m.get("id", "?")
            has_media = m.get("has_media")
            mtype = m.get("media_type", "")
            date_s = (m.get("date") or "")[:16].replace("T", " ")
            text_preview = (m.get("text") or "").replace("\n", " ")[:60]
            icon = "📎" if has_media else "💬"
            if has_media:
                media_count += 1
                text_preview = f"[{mtype}] {text_preview}".strip()
            lines.append(f"`{mid}` {icon} {date_s}\n    {text_preview}")

        lines.append("")
        lines.append(
            f"_Found {media_count} media messages out of {len(history)}._\n"
            "📝 *Send me the message id* of the media you want to download."
        )
        await safe_edit_message_text(
            context.bot, status_msg.chat_id, status_msg.message_id,
            "\n".join(lines),
            reply_markup=kb.cancel_back("mtproto"),
        )
    except mtproto_service.MTProtoError as exc:
        await safe_edit_message_text(
            context.bot, status_msg.chat_id, status_msg.message_id,
            f"❌ `{md_escape(str(exc))}`",
            reply_markup=kb.mtproto_menu(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("mtproto history failed: %s", exc)
        await safe_edit_message_text(
            context.bot, status_msg.chat_id, status_msg.message_id,
            f"❌ Error: `{md_escape(str(exc))}`",
            reply_markup=kb.mtproto_menu(),
        )


async def _mtproto_do_download(update: Update, context: ContextTypes.DEFAULT_TYPE,
                              chat_ref, msg_id: int) -> None:
    """Perform the actual MTProto download and send the file back."""
    status_msg = await safe_reply_text(
        update.message, "📥 Downloading via MTProto…"
    )
    if status_msg is None:
        return

    try:
        path, name, mtype, size = await mtproto_service.download_message_media(
            chat_ref, msg_id,
        )
        caption = (
            f"✅ *Downloaded via MTProto*\n\n"
            f"📄 `{md_escape(name)}`\n"
            f"📦 {human_size(size)}\n"
            f"🗂 {mtype}\n"
            f"🆔 Message: `{msg_id}`"
        )
        if size <= config.upload_limit_bytes:
            with path.open("rb") as fh:
                await safe_send_document(
                    context.bot, update.effective_chat.id, fh,
                    filename=name, caption=caption,
                )
        else:
            await safe_send_message(
                context.bot, update.effective_chat.id,
                caption + "\n\n_ℹ️ Too large to send via Bot API — saved on server._",
            )
        # Auto-delete if configured.
        settings = await repo.ensure_settings(update.effective_user.id)
        if settings.get("auto_delete"):
            await remove_path(path)
        await safe_edit_message_text(
            context.bot, status_msg.chat_id, status_msg.message_id,
            "✅ Done.", reply_markup=kb.mtproto_menu(),
        )
    except mtproto_service.MTProtoError as exc:
        await safe_edit_message_text(
            context.bot, status_msg.chat_id, status_msg.message_id,
            f"❌ `{md_escape(str(exc))}`",
            reply_markup=kb.mtproto_menu(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("mtproto download failed: %s", exc)
        await safe_edit_message_text(
            context.bot, status_msg.chat_id, status_msg.message_id,
            f"❌ Error: `{md_escape(str(exc))}`",
            reply_markup=kb.mtproto_menu(),
        )


def _mtproto_normalize_chat_ref(raw: str) -> str | int | None:
    """Normalize any chat input to a @username string or int chat id."""
    parsed = link_parser.parse_input(raw)
    if not parsed.ok:
        return None
    if parsed.username:
        return f"@{parsed.username}"
    if parsed.chat_id is not None:
        return parsed.chat_id
    return None


def _mtproto_parse_input(raw: str) -> tuple[str | int, int]:
    """Parse inputs that include a message id.

    Supports:
      - https://t.me/channel/123
      - https://t.me/c/1234567890/123
      - @channel 123
      - channel 123

    Raises ``MTProtoError(\"no_message_id\")`` if no message id is present
    (the caller should then show channel history instead).
    """
    parsed = link_parser.parse_input(raw)
    if (parsed.ok and parsed.kind == link_parser.KIND_MESSAGE_LINK
            and parsed.message_id):
        ref = (f"@{parsed.username}" if parsed.username
               else parsed.chat_id)
        return ref, parsed.message_id
    parts = raw.split()
    if len(parts) >= 2:
        ref = parts[0].lstrip("@")
        try:
            return ref, int(parts[-1])
        except ValueError:
            pass
    raise mtproto_service.MTProtoError("no_message_id")


# ===========================================================================
# Handlers — inline
# ===========================================================================


async def inline__handle_inline(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not config.inline_enabled:
        return
    q = update.inline_query
    if q is None:
        return
    user_id = q.from_user.id
    query = (q.query or "").strip()
    is_admin = user_id in config.admin_ids
    await repo.upsert_user(user_id, q.from_user.username, q.from_user.first_name,
                           is_admin)

    parts = query.split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    results = []

    if cmd in ("", "help", "start"):
        results.append(inline__article(
            "help",
            "ℹ️ MediaGrab AI — Help",
            msg.inline_help_text(),
        ))

    if cmd == "info":
        s = await repo.user_stats(user_id)
        results.append(inline__article(
            "info",
            "📊 Your stats",
            msg.inline_stats_text(s),
        ))

    if cmd == "qr" and rest:
        qr_path = await inline__make_qr(rest)
        if qr_path:
            try:
                with qr_path.open("rb") as fh:
                    data = fh.read()
                # Inline mode can't easily return a local file photo without a
                # URL; fall back to a text result instructing the user.
                results.append(inline__article(
                    "qr",
                    f"🔳 QR for: {rest[:40]}",
                    f"Open the bot and use the QR feature to generate this:\n\n`{rest}`",
                ))
            finally:
                await remove_path(qr_path)

    if cmd == "menu":
        results.append(inline__article(
            "menu",
            "🏠 Open MediaGrab AI",
            "Tap to open the bot and use all features.",
            button_text="Open bot",
        ))

    if not results:
        results.append(inline__article(
            "default",
            "🎬 MediaGrab AI Bot",
            msg.inline_help_text(),
        ))

    try:
        await q.answer(results, cache_time=30, is_personal=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("inline answer failed: %s", exc)


def inline__article(_id: str, title: str, description: str,
                    button_text: str | None = None) -> InlineQueryResultArticle:
    content = InputTextMessageContent(message_text=description, parse_mode="Markdown")
    kwargs: dict = {
        "id": _id,
        "title": title,
        "description": description[:200],
        "input_message_content": content,
    }
    if button_text:
        kwargs["reply_markup"] = InlineKeyboardMarkup([[
            InlineKeyboardButton(button_text,
                                 url=f"https://t.me/{config.bot_token.split(':')[0]}")
        ]])
    return InlineQueryResultArticle(**kwargs)


async def inline__make_qr(text: str) -> Path | None:
    try:
        return await qr_generator.generate_qr(text, config.frames_dir)
    except Exception as exc:  # noqa: BLE001
        logger.debug("inline qr failed: %s", exc)
        return None


# ===========================================================================
# Handlers — link_download (Task ID 5)
# ===========================================================================

# Download by Telegram message link or @username — Bot API only.
#
# How it works:
# 1. The user sends a message link, @username, or channel link.
# 2. ``link_parser.parse_input`` classifies the input.
# 3. If a message id is present -> ``copyMessage`` brings the media into the
#    user's chat with the bot and returns a Message with ``file_id``.
# 4. If no message id (channel link / @username) -> ``getChat`` resolves the
#    chat metadata (public only), then we ask the user for a message id.
# 5. The returned ``file_id`` is downloaded via the existing chunked/resumable
#    downloader and sent back as a document with the original filename.
#
# Bot API limitations (surfaced clearly, never silently failing):
#   * Private chats (t.me/c/…) require the bot to be a member.
#   * Public channels can be read via @username + message id.
#   * Bots cannot list messages in a chat — the user must supply the id.
#
# Task ID 6: All user-facing send/edit/reply calls now go through the
# ``safe_*`` wrappers (``safe_reply_text``, ``safe_send_message``,
# ``safe_edit_message_text``) so a Markdown parse error in user-generated text
# (usernames, filenames, captions, chat titles) never raises a
# ``BadRequest: Can't parse entities`` again. Dynamic insertions are escaped
# with ``md_escape``.


class _LinkDownloadError(Exception):
    """Raised when copyMessage/getChat fails with a clear user-facing reason."""


async def link__enter_link_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Called when the user taps '🔗 Download by link or @username'."""
    await states.set_state(update.effective_user.id, states.AWAIT_LINK_DOWNLOAD)
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        text=msg.LINK_DOWNLOAD_PROMPT,
        reply_markup=kb.cancel_back("dl"),
        parse_mode="Markdown",
    )


async def link__handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    raw = (update.message.text or update.message.caption or "").strip()
    if not raw:
        await safe_reply_text(update.message, msg.LINK_DOWNLOAD_EMPTY)
        return

    parsed = link_parser.parse_input(raw)

    if not parsed.ok:
        await safe_reply_text(
            update.message,
            msg.link_parse_error_text(md_escape(parsed.error)),
            reply_markup=kb.cancel_back("dl"),
        )
        return

    # Route by kind.
    if parsed.kind == link_parser.KIND_MESSAGE_LINK:
        await _link_download_message_link(update, context, parsed)
    elif parsed.kind in (link_parser.KIND_CHANNEL_LINK,
                         link_parser.KIND_USERNAME,
                         link_parser.KIND_CHAT_ID):
        await _link_resolve_and_ask_message_id(update, context, parsed)
    else:
        await safe_reply_text(
            update.message,
            msg.link_parse_error_text("Unrecognised input."),
            reply_markup=kb.cancel_back("dl"),
        )


async def _link_download_message_link(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                      parsed: link_parser.ParsedInput) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    from_chat = link_parser.from_chat_reference(parsed)
    msg_id = parsed.message_id

    # Escape the dynamic description so usernames/chat-ids with _ or * don't
    # break Markdown entity parsing.
    description = md_escape(link_parser.describe_input(parsed))
    status = await safe_reply_text(
        update.message,
        msg.link_resolving_text(description),
        reply_markup=kb.cancel_back("dl"),
    )
    if status is None:
        return
    task_id = await repo.create_task(user_id, "download", "running")
    asyncio.create_task(
        _link_run_download(user_id, task_id, chat_id, status.message_id,
                           from_chat, msg_id, parsed, context)
    )


async def _link_run_download(
    user_id: int, task_id: int, chat_id: int, status_msg_id: int,
    from_chat, msg_id: int, parsed: link_parser.ParsedInput,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    path: Path | None = None
    try:
        # 1) copyMessage — brings the media into the bot's accessible scope.
        await _link_edit(context, chat_id, status_msg_id,
                         "📥 Fetching message via copyMessage…")
        copied = await _link_copy_with_retries(context, chat_id, from_chat, msg_id)

        media_type = telegram_media_type(copied)
        if media_type is None:
            raise downloader.DownloadError(
                "That message has no downloadable media."
            )

        # Preserve original filename + caption from the copied message.
        orig_name = _link_extract_filename(copied, media_type)
        caption = copied.caption or copied.text or ""

        # 2) Download the file via getFile + chunked httpx stream.
        await _link_edit(context, chat_id, status_msg_id,
                         "📥 Downloading media…")
        async with media_processor.download_slot():
            path, name, mtype, size = await downloader.download_forwarded(
                copied, progress=None, bot=context.bot,
            )

        await repo.update_task(task_id, status="done", progress=100)

        # 3) Send the file back as a document (preserves original filename).
        #    Escape dynamic filename + caption so they never break Markdown.
        send_caption = msg.download_done_text(md_escape(name), size, mtype)
        if caption:
            send_caption += (
                f"\n\n📝 *Original caption:*\n"
                f"{md_escape(caption[:800])}"
            )
        result = await media_processor.send_file_back(chat_id, path, send_caption, context)

        if result == "too_large":
            await safe_send_message(
                context.bot, chat_id,
                text="ℹ️ The file is too large to send back via the Bot API.",
                reply_markup=kb.main_menu(),
            )
        elif result.startswith("error"):
            await safe_send_message(
                context.bot, chat_id,
                text=(f"⚠️ Could not send the file: "
                      f"`{md_escape(result[6:])}`"),
                reply_markup=kb.main_menu(),
            )

        # 4) Persist to download history.
        await repo.add_download_history(
            user_id, file_name=name, file_size=size, mime_type=None,
            media_type=mtype, source="link", status="done",
            task_id=task_id, message_link=parsed.raw,
        )

        # Stash for Library bookmark + auto-delete.
        await states.set_last_download(user_id, {
            "file_name": name, "media_type": mtype, "file_size": size,
            "file_id": _link_extract_file_id(copied),
        })

        # 5) Clean up the intermediate copied message (keep the chat tidy).
        try:
            await context.bot.delete_message(chat_id=chat_id,
                                             message_id=copied.message_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not delete copied message: %s", exc)

        # Auto-delete the temp file if configured.
        settings = await repo.ensure_settings(user_id)
        if settings.get("auto_delete"):
            await remove_path(path)

        await states.reset(user_id)

    except downloader.FileTooLarge as exc:
        await _link_edit(context, chat_id, status_msg_id,
                         msg.file_too_big_text(config.max_file_size_mb),
                         reply_markup=kb.download_menu())
        await repo.update_task(task_id, status="failed", error=str(exc))
        await repo.add_download_history(
            user_id, file_name=parsed.raw, file_size=0, mime_type=None,
            media_type="link", source="link", status="failed",
            task_id=task_id, message_link=parsed.raw,
        )
    except _LinkDownloadError as exc:
        await _link_edit(context, chat_id, status_msg_id,
                         msg.link_download_failed_text(md_escape(str(exc))),
                         reply_markup=kb.download_menu())
        await repo.update_task(task_id, status="failed", error=str(exc))
        await repo.add_download_history(
            user_id, file_name=parsed.raw, file_size=0, mime_type=None,
            media_type="link", source="link", status="failed",
            task_id=task_id, message_link=parsed.raw,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("link download failed: %s", exc)
        await _link_edit(context, chat_id, status_msg_id,
                         msg.link_download_failed_text(md_escape(str(exc))),
                         reply_markup=kb.download_menu())
        await repo.update_task(task_id, status="failed", error=str(exc))
        await repo.add_download_history(
            user_id, file_name=parsed.raw, file_size=0, mime_type=None,
            media_type="link", source="link", status="failed",
            task_id=task_id, message_link=parsed.raw,
        )
    finally:
        if path and path.exists():
            try:
                await remove_path(path)
            except Exception:  # noqa: BLE001
                pass


async def _link_copy_with_retries(context, chat_id, from_chat, msg_id):
    """Call copyMessage with smart retries.

    Retries ONLY on transient errors (RetryAfter, NetworkError, TimedOut).
    Permanent errors (BadRequest \"not found\", Forbidden, content-protected)
    fail immediately with a clear, specific user-facing message — no retry.
    """
    max_retries = config.max_retries
    base_delay = 2.0
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return await context.bot.copy_message(
                chat_id=chat_id,
                from_chat_id=from_chat,
                message_id=msg_id,
            )
        except RetryAfter as exc:
            # Rate-limited — sleep and retry (this IS retriable).
            wait = min(exc.retry_after + 1, 30)
            logger.info("Rate limited on copyMessage: %ss (attempt %d/%d)",
                        exc.retry_after, attempt, max_retries)
            await asyncio.sleep(wait)
            last_exc = exc
            continue
        except BadRequest as exc:
            # BadRequest is PERMANENT — never retry. Map to a clear message.
            raise _link_map_copy_bad_request(exc, from_chat, msg_id) from exc
        except Forbidden as exc:
            # Forbidden is PERMANENT — never retry.
            exc_text = str(exc).lower()
            if "not a member" in exc_text or "bot is not a member" in exc_text:
                raise _LinkDownloadError(
                    "The bot is not a member of that chat. For private "
                    "channels or groups, you must add the bot as a member "
                    "first (or make it an admin)."
                ) from exc
            raise _LinkDownloadError(
                "The bot doesn't have permission to read that chat."
            ) from exc
        except (NetworkError, TimedOut) as exc:
            # Transient — retry with back-off.
            last_exc = exc
            if attempt < max_retries:
                delay = min(base_delay * (2 ** (attempt - 1)), 30.0)
                logger.info("Network error on copyMessage (attempt %d/%d), "
                            "retrying in %.1fs: %s", attempt, max_retries, delay, exc)
                await asyncio.sleep(delay)
                continue
            raise _LinkDownloadError(
                f"Network error after {max_retries} retries: {exc}"
            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Unknown error — don't retry, surface it.
            raise _LinkDownloadError(f"Unexpected error: {exc}") from exc

    # Exhausted retries on a transient error.
    raise _LinkDownloadError(
        f"Request failed after {max_retries} retries: {last_exc}"
    )


def _link_map_copy_bad_request(exc: BadRequest, from_chat, msg_id: int) -> _LinkDownloadError:
    """Map a copyMessage BadRequest to the most helpful user-facing message."""
    msg_text = str(exc).lower()

    # Content-protected channel (restricted content).
    # Telegram blocks copying/saving from protected channels. The error
    # message is often "message to copy not found" even though the message
    # exists — this is intentional obfuscation by Telegram.
    if ("message to copy not found" in msg_text
            or "message not found" in msg_text
            or "message to forward not found" in msg_text):
        return _LinkDownloadError(
            f"❌ Cannot download message #{msg_id}.\n\n"
            "This is usually one of:\n"
            "• The channel has *content protection* enabled (restricted "
            "content) — Telegram blocks bots from copying/saving these.\n"
            "• The message was deleted or doesn't exist.\n"
            "• The message id is wrong.\n\n"
            "_Content-protected channels cannot be downloaded by any bot — "
            "this is a Telegram platform limitation._"
        )

    # Chat not accessible.
    if "chat not found" in msg_text:
        return _LinkDownloadError(
            "The bot cannot access that chat. This means either:\n"
            "• The chat doesn't exist or the username is wrong\n"
            "• It's a private chat and the bot hasn't been added as a member\n"
            "• The channel has restricted who can view its content"
        )

    # Bot blocked by user.
    if "bot was blocked by the user" in msg_text:
        return _LinkDownloadError(
            "The bot was blocked by the user. Please /start the bot first."
        )

    # Content forwarding restricted.
    if "forwarding" in msg_text and "restrict" in msg_text:
        return _LinkDownloadError(
            "This channel has *restricted forwarding* — Telegram prevents "
            "bots from copying or downloading its content."
        )

    # Generic fallback.
    return _LinkDownloadError(f"Telegram rejected the request: {exc}")


async def _link_resolve_and_ask_message_id(update: Update,
                                           context: ContextTypes.DEFAULT_TYPE,
                                           parsed: link_parser.ParsedInput) -> None:
    user_id = update.effective_user.id
    status = await safe_reply_text(update.message, "🔍 Resolving chat…")
    if status is None:
        return

    # If MTProto is available and the input is a @username or channel link,
    # route to the media browser (category browsing + Download All).
    if mtproto_manager.is_started() and (parsed.username or parsed.chat_id):
        chat_ref = (f"@{parsed.username}" if parsed.username
                    else parsed.chat_id)
        # Delete the "Resolving…" status and hand off to the media browser.
        try:
            await context.bot.delete_message(chat_id=status.chat_id,
                                             message_id=status.message_id)
        except Exception:  # noqa: BLE001
            pass
        await media_browser_handler.show_categories(update, context, chat_ref)
        return

    # For public usernames, resolve via getChat to confirm + show info.
    if parsed.username:
        try:
            chat = await _link_get_chat_with_retries(context, f"@{parsed.username}")
        except _LinkDownloadError as exc:
            await _link_edit(context, status.chat_id, status.message_id,
                             msg.link_download_failed_text(md_escape(str(exc))),
                             reply_markup=kb.download_menu())
            return
        info = _link_chat_info_dict(chat)
        # Stash the resolved reference for the message-id step.
        context.user_data["link_from_chat"] = f"@{parsed.username}"
        context.user_data["link_chat_info"] = info
        await _link_edit(context, status.chat_id, status.message_id,
                         msg.link_resolved_ask_msgid(_link_escape_info(info)),
                         reply_markup=kb.cancel_back("dl"))
    elif parsed.chat_id is not None:
        # Raw chat id or private channel link — try getChat.
        try:
            chat = await _link_get_chat_with_retries(context, parsed.chat_id)
            info = _link_chat_info_dict(chat)
        except _LinkDownloadError as exc:
            # Even if getChat fails, we can still try copyMessage with the id.
            info = {"title": str(parsed.chat_id), "type": "chat",
                    "chat_id": parsed.chat_id}
        context.user_data["link_from_chat"] = parsed.chat_id
        context.user_data["link_chat_info"] = info
        await _link_edit(context, status.chat_id, status.message_id,
                         msg.link_resolved_ask_msgid(_link_escape_info(info)),
                         reply_markup=kb.cancel_back("dl"))
    else:
        await _link_edit(context, status.chat_id, status.message_id,
                         msg.link_parse_error_text("Cannot resolve that input."),
                         reply_markup=kb.download_menu())

    # Switch to the message-id-awaiting state.
    await states.set_state(user_id, states.AWAIT_LINK_MESSAGE_ID)


async def link__handle_message_id_input(update: Update,
                                        context: ContextTypes.DEFAULT_TYPE) -> None:
    """User sent a message id after we resolved a channel/username."""
    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    try:
        msg_id = int(text)
    except ValueError:
        await safe_reply_text(
            update.message,
            "⚠️ Please send a numeric message id (e.g. `123`). "
            "You can find it in the original t.me link.",
            reply_markup=kb.cancel_back("dl"),
        )
        return
    if msg_id <= 0:
        await safe_reply_text(
            update.message,
            "⚠️ Message id must be a positive number.",
            reply_markup=kb.cancel_back("dl"),
        )
        return

    from_chat = context.user_data.get("link_from_chat")
    if not from_chat:
        await safe_reply_text(
            update.message,
            "⚠️ Session expired. Please send the link or @username again.",
            reply_markup=kb.download_menu(),
        )
        await states.reset(user_id)
        return

    # Build a synthetic ParsedInput and reuse the message-link path.
    parsed = link_parser.ParsedInput(
        kind=link_parser.KIND_MESSAGE_LINK,
        visibility=link_parser.VIS_PUBLIC if isinstance(from_chat, str) else link_parser.VIS_PRIVATE,
        username=from_chat.lstrip("@") if isinstance(from_chat, str) else None,
        chat_id=from_chat if isinstance(from_chat, int) else None,
        message_id=msg_id,
        raw=f"{from_chat}/{msg_id}",
    )
    await _link_download_message_link(update, context, parsed)


async def _link_get_chat_with_retries(context, chat_ref):
    """Resolve a chat via getChat with SMART retries.

    Retries ONLY on transient errors (RetryAfter, NetworkError, TimedOut).
    Permanent errors (BadRequest \"not found\", Forbidden) fail immediately —
    no wasted retries.
    """
    max_retries = min(config.max_retries, 3)
    base_delay = 2.0
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return await context.bot.get_chat(chat_ref)
        except RetryAfter as exc:
            wait = min(exc.retry_after + 1, 30)
            logger.info("Rate limited on getChat: %ss (attempt %d/%d)",
                        exc.retry_after, attempt, max_retries)
            await asyncio.sleep(wait)
            last_exc = exc
            continue
        except BadRequest as exc:
            # Permanent — don't retry.
            msg_text = str(exc).lower()
            logger.info("get_chat failed for %s: %s", chat_ref, exc)
            if "chat not found" in msg_text:
                raise _LinkDownloadError(
                    f"Chat {chat_ref} was not found. It may not exist, is "
                    "private, or the bot cannot access it."
                ) from exc
            raise _LinkDownloadError(f"Telegram rejected getChat: {exc}") from exc
        except Forbidden as exc:
            raise _LinkDownloadError(
                "The bot doesn't have permission to access that chat. "
                "Private chats require the bot to be a member."
            ) from exc
        except (NetworkError, TimedOut) as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = min(base_delay * (2 ** (attempt - 1)), 30.0)
                logger.info("Network error on getChat (attempt %d/%d), "
                            "retrying in %.1fs", attempt, max_retries, delay)
                await asyncio.sleep(delay)
                continue
            raise _LinkDownloadError(
                f"Network error after {max_retries} retries: {exc}"
            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _LinkDownloadError(f"Failed to resolve chat: {exc}") from exc

    raise _LinkDownloadError(
        f"Request failed after {max_retries} retries: {last_exc}"
    )


def _link_chat_info_dict(chat) -> dict:
    return {
        "title": getattr(chat, "title", None),
        "type": getattr(chat, "type", None),
        "username": getattr(chat, "username", None),
        "first_name": getattr(chat, "first_name", None),
        "last_name": getattr(chat, "last_name", None),
        "chat_id": getattr(chat, "id", None),
        "members": getattr(chat, "member_count", None),
        "description": getattr(chat, "description", None),
        "bio": getattr(chat, "bio", None),
    }


def _link_escape_info(info: dict) -> dict:
    """Return a copy of *info* with all string values Markdown-escaped."""
    out = {}
    for k, v in info.items():
        if isinstance(v, str):
            out[k] = md_escape(v)
        else:
            out[k] = v
    return out


def _link_extract_filename(message, media_type: str) -> str | None:
    """Pull the original filename from a PTB message, if available."""
    for attr in ("document", "video", "audio", "animation",
                 "voice", "video_note"):
        cand = getattr(message, attr, None)
        if cand is not None and getattr(cand, "file_name", None):
            return cand.file_name
    if message.photo:
        return None  # photos have no filename
    return None


def _link_extract_file_id(message) -> str | None:
    for attr in ("document", "video", "audio", "animation",
                 "voice", "video_note"):
        cand = getattr(message, attr, None)
        if cand is not None and getattr(cand, "file_id", None):
            return cand.file_id
    if message.photo:
        return message.photo[-1].file_id
    return None


async def _link_edit(context, chat_id: int, message_id: int, text: str,
                     reply_markup=None) -> None:
    """Safe edit — tries Markdown, falls back to plain text on parse error."""
    await safe_edit_message_text(
        context.bot, chat_id, message_id, text,
        reply_markup=reply_markup,
    )


# ===========================================================================
# Handlers — media_browser (channel category browsing + Download All)
# ===========================================================================
# Activated when a user enters a @username or channel link (without a message
# id) in the Download menu and MTProto is available. Flow:
#
#   1. User sends `@channel` in link-download mode
#   2. Bot resolves via MTProto, scans recent history, shows media categories
#   3. User taps a category (e.g. "🎬 Videos (12)")
#   4. Bot shows the list of items + a "Download All" button
#   5. User taps "Download All" or taps an individual item id
#   6. Bot downloads via MTProto with progress, sends files back
#
# This runs on top of the existing MTProto backend; the Bot API handles all UX.
# ===========================================================================


async def mb__show_categories(update: Update, context: ContextTypes.DEFAULT_TYPE,
                              chat_ref) -> None:
    """Resolve a channel via MTProto, scan history, show media categories."""
    chat_id = update.effective_chat.id

    status_msg = await safe_reply_text(
        update.message, "📂 Scanning channel media via MTProto…"
    )
    if status_msg is None:
        return

    try:
        # Resolve + scan.
        info = await mtproto_service.resolve_entity(chat_ref)
        by_cat = await media_browser.scan_channel_media(chat_ref, limit=100)
    except mtproto_service.MTProtoError as exc:
        await safe_edit_message_text(
            context.bot, status_msg.chat_id, status_msg.message_id,
            f"❌ `{md_escape(str(exc))}`",
            reply_markup=kb.download_menu(),
        )
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("media scan failed: %s", exc)
        await safe_edit_message_text(
            context.bot, status_msg.chat_id, status_msg.message_id,
            f"❌ Error: `{md_escape(str(exc))}`",
            reply_markup=kb.download_menu(),
        )
        return

    # Stash for the category/menu steps.
    context.user_data["mb_chat_ref"] = chat_ref
    context.user_data["mb_by_cat"] = by_cat
    context.user_data["mb_chat_info"] = info

    title = md_escape(info.get("title") or info.get("first_name")
                      or str(info.get("id", chat_ref)))
    uname = info.get("username")
    uname_s = f"@{md_escape(uname)}" if uname else "—"
    summary = media_browser.scan_summary(by_cat)

    # Build a keyboard with one button per non-empty category.
    rows: list[list[InlineKeyboardButton]] = []
    for cat_key in media_browser.CATEGORY_KEYS:
        if cat_key in by_cat:
            count = len(by_cat[cat_key])
            label = f"{media_browser.category_label(cat_key)} ({count})"
            rows.append([InlineKeyboardButton(label, callback_data=f"mb:cat:{cat_key}")])
    rows.append([InlineKeyboardButton("🔙 Back to Download", callback_data="b:dl")])
    markup = InlineKeyboardMarkup(rows)

    text = (
        f"📂 *Channel media browser*\n\n"
        f"📛 *Name:* {title}\n"
        f"🔗 *Username:* {uname_s}\n\n"
        f"{summary}\n\n"
        f"_Tap a category to browse and download._"
    )
    await safe_edit_message_text(
        context.bot, status_msg.chat_id, status_msg.message_id,
        text, reply_markup=markup,
    )


async def mb__show_category_items(update: Update,
                                  context: ContextTypes.DEFAULT_TYPE,
                                  category: str) -> None:
    """Show the list of media items in a category with a Download All button."""
    by_cat = context.user_data.get("mb_by_cat") or {}
    items = by_cat.get(category, [])
    if not items:
        await update.callback_query.answer("No items in this category.")
        return

    await update.callback_query.answer()
    label = media_browser.category_label(category)
    lines = [f"{label} — *{len(items)} items*", ""]
    for i, item in enumerate(items[:20], 1):  # show first 20
        mid = item.get("message_id", "?")
        date_s = (item.get("date") or "")[:10]
        text_prev = (item.get("text") or "").replace("\n", " ")[:40]
        lines.append(f"`{mid}` · {date_s} · {text_prev}")
    if len(items) > 20:
        lines.append(f"\n_…and {len(items) - 20} more. Use Download All._")

    rows: list[list[InlineKeyboardButton]] = []
    # Individual item buttons (first 10 for quick access).
    for item in items[:10]:
        mid = item.get("message_id")
        rows.append([InlineKeyboardButton(
            f"#{mid}", callback_data=f"mb:one:{mid}"
        )])
    rows.append([InlineKeyboardButton(
        f"⬇️ Download All ({len(items)})", callback_data=f"mb:all:{category}"
    )])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="mb:back")])
    markup = InlineKeyboardMarkup(rows)

    await update.callback_query.edit_message_text(
        text="\n".join(lines),
        reply_markup=markup,
        parse_mode="Markdown",
    )


async def mb__download_single(update: Update, context: ContextTypes.DEFAULT_TYPE,
                              message_id: int) -> None:
    chat_ref = context.user_data.get("mb_chat_ref")
    if not chat_ref:
        await update.callback_query.answer("Session expired. Restart.")
        return
    await update.callback_query.answer("Downloading…")
    status_msg_id = update.callback_query.message.message_id
    chat_id = update.effective_chat.id

    await safe_edit_message_text(
        context.bot, chat_id, status_msg_id,
        f"📥 Downloading message #{message_id} via MTProto…",
    )
    result = await media_browser.download_one(chat_ref, message_id)
    if result.get("ok"):
        path = result["path"]
        name = result["filename"]
        size = result["size"]
        mtype = result["media_type"]
        caption = (
            f"✅ *Downloaded via MTProto*\n\n"
            f"📄 `{md_escape(name)}`\n"
            f"📦 {human_size(size)}\n"
            f"🗂 {mtype}\n"
            f"🆔 Message: `{message_id}`"
        )
        if size <= config.upload_limit_bytes:
            with path.open("rb") as fh:
                await safe_send_document(
                    context.bot, chat_id, fh,
                    filename=name, caption=caption,
                )
        else:
            await safe_send_message(
                context.bot, chat_id,
                caption + "\n\n_ℹ️ Too large for Bot API — saved on server._",
            )
        # Auto-delete if configured.
        settings = await repo.ensure_settings(update.effective_user.id)
        if settings.get("auto_delete"):
            await remove_path(path)
        await safe_edit_message_text(
            context.bot, chat_id, status_msg_id,
            "✅ Done.", reply_markup=kb.mtproto_menu(),
        )
    else:
        await safe_edit_message_text(
            context.bot, chat_id, status_msg_id,
            f"❌ `{md_escape(result.get('error', 'unknown'))}`",
            reply_markup=kb.mtproto_menu(),
        )


async def mb__download_all(update: Update, context: ContextTypes.DEFAULT_TYPE,
                           category: str) -> None:
    chat_ref = context.user_data.get("mb_chat_ref")
    by_cat = context.user_data.get("mb_by_cat") or {}
    items = by_cat.get(category, [])
    if not chat_ref or not items:
        await update.callback_query.answer("Session expired or no items.")
        return

    await update.callback_query.answer(f"Downloading {len(items)} items…")
    chat_id = update.effective_chat.id
    status_msg_id = update.callback_query.message.message_id

    label = media_browser.category_label(category)
    dedup: set[int] = set()

    async def progress_cb(done, total, msg_id, ok):
        pct = done / total * 100 if total else 0
        await safe_edit_message_text(
            context.bot, chat_id, status_msg_id,
            f"⬇️ *{label} bulk download*\n\n"
            f"Progress: {done}/{total} ({pct:.0f}%)\n"
            f"✅ Sending files as they complete…",
        )

    async def item_done_cb(result):
        if result.get("ok"):
            path = result["path"]
            name = result["filename"]
            size = result["size"]
            if size <= config.upload_limit_bytes:
                caption = (
                    f"✅ `{md_escape(name)}`\n"
                    f"📦 {human_size(size)} · 🆔 `{result['message_id']}`"
                )
                try:
                    with path.open("rb") as fh:
                        await safe_send_document(
                            context.bot, chat_id, fh,
                            filename=name, caption=caption,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("send failed for #%s: %s",
                                   result["message_id"], exc)
            # Auto-delete if configured.
            settings = await repo.ensure_settings(update.effective_user.id)
            if settings.get("auto_delete"):
                await remove_path(path)

    bulk_result = await media_browser.download_category(
        chat_ref, category, items,
        progress_cb=progress_cb,
        item_done_cb=item_done_cb,
        dedup=dedup,
    )
    await safe_edit_message_text(
        context.bot, chat_id, status_msg_id,
        media_browser.bulk_summary(bulk_result, category),
        reply_markup=kb.mtproto_menu(),
    )


async def mb__back_to_categories(update: Update,
                                 context: ContextTypes.DEFAULT_TYPE) -> None:
    by_cat = context.user_data.get("mb_by_cat") or {}
    info = context.user_data.get("mb_chat_info") or {}
    chat_ref = context.user_data.get("mb_chat_ref", "channel")
    await update.callback_query.answer()
    title = md_escape(info.get("title") or info.get("first_name")
                      or str(info.get("id", chat_ref)))
    uname = info.get("username")
    uname_s = f"@{md_escape(uname)}" if uname else "—"
    summary = media_browser.scan_summary(by_cat)

    rows: list[list[InlineKeyboardButton]] = []
    for cat_key in media_browser.CATEGORY_KEYS:
        if cat_key in by_cat:
            count = len(by_cat[cat_key])
            label = f"{media_browser.category_label(cat_key)} ({count})"
            rows.append([InlineKeyboardButton(label, callback_data=f"mb:cat:{cat_key}")])
    rows.append([InlineKeyboardButton("🔙 Back to Download", callback_data="b:dl")])
    markup = InlineKeyboardMarkup(rows)

    await update.callback_query.edit_message_text(
        text=(f"📂 *Channel media browser*\n\n"
              f"📛 *Name:* {title}\n"
              f"🔗 *Username:* {uname_s}\n\n"
              f"{summary}\n\n_Tap a category to browse and download._"),
        reply_markup=markup,
        parse_mode="Markdown",
    )


# ===========================================================================
# Handlers — vc_admin (Task ID 12)
# ===========================================================================
# Admin-only controls for the VC tour subsystem. Accessible from the existing
# admin menu via a new "🎙 VC Control" button. All actions delegate to the
# ``vc_tour`` namespace defined in the Services section above.
# ===========================================================================


def _vc_is_admin(user_id: int) -> bool:
    return user_id in config.vc_admin_ids


def _vc_mtproto_ok() -> bool:
    return mtproto_is_started()


def _vc_status_text(s: dict) -> str:
    running = "▶️ Running" if s.get("running") else "⏹ Stopped"
    paused = " (Paused)" if s.get("paused") else ""
    current = "—"
    if s.get("current_group_id"):
        title = md_escape(s.get("current_title") or "Unknown")
        current = f"{title} (`{s['current_group_id']}`)"
    mode = s.get("mode", "auto")
    visited = s.get("visited_count", 0)
    queue = f"{s.get('queue_index', 0)}/{s.get('queue_size', 0)}"
    return (
        f"*🎙 VC Tour Control*\n\n"
        f"Status: {running}{paused}\n"
        f"Current VC: {current}\n"
        f"Mode: {mode}\n"
        f"Queue: {queue}\n"
        f"Visited this tour: {visited}\n"
        f"Stay: {config.vc_stay_minutes} min · Cooldown: {config.vc_cooldown_seconds}s\n\n"
        f"_MTProto: {'✅' if _vc_mtproto_ok() else '❌ not started'}_"
    )


async def vc__show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer()
        await safe_edit_message_text(
            context.bot, update.effective_chat.id,
            update.callback_query.message.message_id,
            "🚫 Admin only.", reply_markup=back_only("admin"),
        )
        return
    await update.callback_query.answer()
    status = vc_get_status()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        _vc_status_text(status),
        reply_markup=vc_menu(),
    )


# --- Tour controls ----------------------------------------------------------

async def vc__start_tour(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    if not _vc_mtproto_ok():
        await update.callback_query.answer("MTProto not started.", show_alert=True)
        return
    await update.callback_query.answer("Starting tour…")
    result = await vc_start_tour()
    status = vc_get_status()
    text = f"*VC Tour*\n\nResult: `{md_escape(result)}`\n\n" + _vc_status_text(status)
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        text, reply_markup=vc_menu(),
    )


async def vc__pause_tour(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer("Pausing…")
    result = await vc_pause_tour()
    status = vc_get_status()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        f"*VC Tour*\n\nResult: `{md_escape(result)}`\n\n" + _vc_status_text(status),
        reply_markup=vc_menu(),
    )


async def vc__resume_tour(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer("Resuming…")
    result = await vc_resume_tour()
    status = vc_get_status()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        f"*VC Tour*\n\nResult: `{md_escape(result)}`\n\n" + _vc_status_text(status),
        reply_markup=vc_menu(),
    )


async def vc__stop_tour(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer("Stopping…")
    result = await vc_stop_tour()
    status = vc_get_status()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        f"*VC Tour*\n\nResult: `{md_escape(result)}`\n\n" + _vc_status_text(status),
        reply_markup=vc_menu(),
    )


async def vc__refresh_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer()
    status = vc_get_status()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        _vc_status_text(status), reply_markup=vc_menu(),
    )


async def vc__refresh_groups(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    if not _vc_mtproto_ok():
        await update.callback_query.answer("MTProto not started.", show_alert=True)
        return
    await update.callback_query.answer("Discovering groups…")
    result = await vc_discover_groups()
    text = (
        f"*🔄 Group Discovery*\n\n"
        f"Discovered: {result.get('discovered', 0)}\n"
        f"Skipped: {result.get('skipped', 0)}\n"
        f"Errors: {result.get('errors', 0)}\n\n"
        + _vc_status_text(vc_get_status())
    )
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        text, reply_markup=vc_menu(),
    )


# --- Current VC status ------------------------------------------------------

async def vc__show_current(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer()
    s = vc_get_status()
    if not s.get("current_group_id"):
        text = "🎙 *Current Voice Chat*\n\nNo active VC session."
    else:
        elapsed = s.get("elapsed_seconds") or 0
        link = s.get("current_link") or "Private / no shareable link available"
        members = s.get("current_members")
        members_s = f"{members:,}" if members else "Unavailable"
        text = (
            "🎙 *Current Voice Chat*\n\n"
            f"🏷 *Group:* {md_escape(s.get('current_title') or 'Unknown')}\n"
            f"🆔 *Group ID:* `{s.get('current_group_id')}`\n"
            f"🔗 *Link:* {md_escape(link)}\n"
            f"👥 *Members:* {members_s}\n"
            f"🕒 *Joined At:* `{s.get('joined_at_iso', '—')}`\n"
            f"⏱ *Elapsed:* {_vc_duration_str(elapsed)}\n"
            f"⏳ *Planned:* {config.vc_stay_minutes} min\n"
            f"🎯 *Mode:* {s.get('mode')}\n"
            f"📡 *Status:* {'Connected' if s.get('current_group_id') else 'Disconnected'}\n"
            f"📍 *Queue:* {s.get('queue_index', 0)}/{s.get('queue_size', 0)}\n"
            f"▶️ *Tour:* {'Running' if s.get('running') else 'Stopped'}"
            f"{' (Paused)' if s.get('paused') else ''}"
        )
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        text, reply_markup=vc_menu(),
    )


# --- History ----------------------------------------------------------------

async def vc__show_history(update: Update, context: ContextTypes.DEFAULT_TYPE,
                           page: int = 0) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer()
    limit = 10
    offset = page * limit
    visits = await vc_recent_visits(limit=limit, offset=offset)
    total = await vc_visit_count()
    if not visits:
        text = "📜 *VC History*\n\nNo visits recorded yet."
    else:
        lines = [f"📜 *VC History* ({total} total, page {page + 1})", ""]
        for v in visits:
            title = md_escape(v.get("group_title") or "Unknown")
            gid = v.get("group_id", "?")
            joined = (v.get("joined_at") or "")[:16].replace("T", " ")
            left = (v.get("left_at") or "")[:16].replace("T", " ")
            dur = v.get("actual_duration_seconds")
            dur_s = f"{dur}s" if dur else "—"
            mode = v.get("mode", "?")
            status = v.get("status", "?")
            reason = v.get("leave_reason") or ""
            icon = {"completed": "✅", "failed": "❌", "joined": "🎙",
                    "disconnected": "⚠️"}.get(status, "•")
            lines.append(
                f"{icon} `{joined}` → `{left}`\n"
                f"    {title} (`{gid}`)\n"
                f"    {mode} · {dur_s} · {status}"
                + (f" · {reason}" if reason else "")
            )
        text = "\n".join(lines)
    # Pagination buttons.
    rows: list[list[InlineKeyboardButton]] = []
    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️ Prev", callback_data=f"vc:hist:{page - 1}"))
    if offset + limit < total:
        nav.append(InlineKeyboardButton("➡️ Next", callback_data=f"vc:hist:{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="b:vc")])
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        text, reply_markup=InlineKeyboardMarkup(rows),
    )


# --- Manual VC control ------------------------------------------------------

async def vc__show_manual_menu(update: Update,
                               context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        "*🎯 Manual VC Control*\n\n"
        "Tap an action below. For *Join Target VC*, you'll be asked to send "
        "a @groupusername, link, or numeric group id.",
        reply_markup=vc_manual_menu(),
    )


async def vc__enter_join_target(update: Update,
                               context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await set_state(update.effective_user.id, AWAIT_VC_JOIN_TARGET)
    await update.callback_query.answer()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        "🎙 *Join Target VC*\n\n"
        "Send me the @groupusername, public group link, or accessible numeric "
        "group ID of the group whose voice chat you want to join.",
        reply_markup=cancel_back("vcmanual"),
    )


async def vc__handle_join_target(update: Update,
                                context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await safe_reply_text(update.message, "🚫 Not authorised.")
        return
    raw = (update.message.text or "").strip()
    if not raw:
        await safe_reply_text(update.message, "⚠️ Send a group reference.")
        return
    await reset(update.effective_user.id)
    status_msg = await safe_reply_text(update.message, "🎙 Joining target VC…")
    if status_msg is None:
        return
    result = await vc_manual_join(raw)
    await safe_edit_message_text(
        context.bot, status_msg.chat_id, status_msg.message_id,
        f"*Manual VC Join*\n\nResult: `{md_escape(result)}`",
        reply_markup=vc_manual_menu(),
    )


async def vc__leave_current(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer("Leaving…")
    result = await vc_manual_leave()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        f"*Manual VC Leave*\n\nResult: `{md_escape(result)}`",
        reply_markup=vc_manual_menu(),
    )


async def vc__check_target(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await set_state(update.effective_user.id, AWAIT_VC_CHECK_TARGET)
    await update.callback_query.answer()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        "🔎 *Check Target VC*\n\nSend me a @groupusername, link, or numeric id "
        "to check if it has an active voice chat.",
        reply_markup=cancel_back("vcmanual"),
    )


async def vc__handle_check_target(update: Update,
                                 context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await safe_reply_text(update.message, "🚫 Not authorised.")
        return
    raw = (update.message.text or "").strip()
    await reset(update.effective_user.id)
    status_msg = await safe_reply_text(update.message, "🔎 Checking…")
    if status_msg is None:
        return
    client = mtproto_get_client()
    if client is None:
        await safe_edit_message_text(
            context.bot, status_msg.chat_id, status_msg.message_id,
            "❌ MTProto not started.",
            reply_markup=vc_manual_menu(),
        )
        return
    try:
        entity = await client.get_entity(raw)
        info = await vc_detect_active_call(client, entity)
        if info is None:
            text = "ℹ️ No active voice chat in that group."
        else:
            members = info.get("members")
            members_s = f"{members:,}" if members else "Unavailable"
            text = (f"✅ *Active VC found!*\n\n"
                    f"🆔 Call ID: `{info['call_id']}`\n"
                    f"👥 Members: {members_s}")
    except Exception as exc:  # noqa: BLE001
        text = f"❌ `{md_escape(str(exc))}`"
    await safe_edit_message_text(
        context.bot, status_msg.chat_id, status_msg.message_id,
        text, reply_markup=vc_manual_menu(),
    )


async def vc__stay_5min(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer()
    result = await vc_set_stay_duration(5)
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        f"*Stay Duration*\n\nResult: `{md_escape(result)}`",
        reply_markup=vc_manual_menu(),
    )


async def vc__enter_custom_stay(update: Update,
                               context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await set_state(update.effective_user.id, AWAIT_VC_STAY)
    await update.callback_query.answer()
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        f"⏱ *Custom Stay Duration*\n\n"
        f"Send a number of minutes ({config.vc_min_stay_minutes}–{config.vc_max_stay_minutes}).",
        reply_markup=cancel_back("vcmanual"),
    )


async def vc__handle_custom_stay(update: Update,
                                context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await safe_reply_text(update.message, "🚫 Not authorised.")
        return
    text = (update.message.text or "").strip()
    try:
        minutes = int(text)
    except ValueError:
        await safe_reply_text(update.message, "⚠️ Send a numeric value.")
        return
    await reset(update.effective_user.id)
    result = await vc_set_stay_duration(minutes)
    await safe_send_message(
        context.bot, update.effective_chat.id,
        f"*Stay Duration*\n\nResult: `{md_escape(result)}`",
        reply_markup=vc_manual_menu(),
    )


# --- Settings ---------------------------------------------------------------

async def vc__show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _vc_is_admin(update.effective_user.id):
        await update.callback_query.answer("Not authorised.", show_alert=True)
        return
    await update.callback_query.answer()
    text = (
        "*⚙️ VC Settings*\n\n"
        f"⏱ Stay: {config.vc_stay_minutes} min\n"
        f"⏳ Cooldown: {config.vc_cooldown_seconds}s\n"
        f"🔁 Revisit: {'On' if config.vc_revisit_same_group else 'Off'}\n"
        f"▶️ Auto-resume after manual: {'On' if config.vc_auto_resume_after_manual else 'Off'}\n"
        f"🔔 Join notifications: {'On' if config.vc_join_notifications else 'Off'}\n"
        f"🔔 Leave notifications: {'On' if config.vc_leave_notifications else 'Off'}\n"
        f"📊 Save history: {'On' if config.vc_save_history else 'Off'}\n\n"
        "_Settings are read from .env. Edit .env and restart to change._"
    )
    await safe_edit_message_text(
        context.bot, update.effective_chat.id,
        update.callback_query.message.message_id,
        text, reply_markup=vc_settings_menu(),
    )


# ===========================================================================
# Handler namespace aliases (so menu.py code works unchanged)
# ===========================================================================

download_handler = SimpleNamespace(
    enter_forward_mode=download__enter_forward_mode,
    handle_forwarded_media=download__handle_forwarded_media,
)

analyze_handler = SimpleNamespace(
    enter_analyze_mode=analyze__enter_analyze_mode,
    show_modes=analyze__show_modes,
    set_mode=analyze__set_mode,
    handle_forwarded_video=analyze__handle_forwarded_video,
)

history_handler = SimpleNamespace(
    show_downloads=history__show_downloads,
    show_ai=history__show_ai,
    clear=history__clear,
)

settings_handler = SimpleNamespace(
    handle_setting=settings__handle_setting,
    handle_gemini_key_input=settings__handle_gemini_key_input,
)

help_handler = SimpleNamespace(
    show_section=help__show_section,
)

inspector_handler = SimpleNamespace(
    enter_inspect_mode=inspector__enter_inspect_mode,
    show_recent=inspector__show_recent,
    handle_inspect_input=inspector__handle_inspect_input,
)

toolbox_handler = SimpleNamespace(
    handle_tool_selection=toolbox__handle_tool_selection,
    handle_forwarded_media=toolbox__handle_forwarded_media,
    handle_imgconv_format=toolbox__handle_imgconv_format,
)

library_handler = SimpleNamespace(
    show_menu=library__show_menu,
    handle_action=library__handle_action,
    handle_search_input=library__handle_search_input,
)

stats_handler = SimpleNamespace(
    show_menu=stats__show_menu,
    show_user=stats__show_user,
    show_global=stats__show_global,
)

qr_handler = SimpleNamespace(
    show_menu=qr__show_menu,
    enter_make_mode=qr__enter_make_mode,
    handle_input=qr__handle_input,
)

batch_handler = SimpleNamespace(
    enter_batch_mode=batch__enter_batch_mode,
    handle_batch_message=batch__handle_batch_message,
)

scheduled_handler = SimpleNamespace(
    show_menu=scheduled__show_menu,
    show_list=scheduled__show_list,
    handle_schedule_input=scheduled__handle_schedule_input,
    queue_scheduled=scheduled__queue_scheduled,
    run_due_tasks=scheduled__run_due_tasks,
)

backup_handler = SimpleNamespace(
    show_menu=backup__show_menu,
    do_export=backup__do_export,
    enter_import=backup__enter_import,
    handle_import_file=backup__handle_import_file,
)

admin_handler = SimpleNamespace(
    show_menu=admin__show_menu,
    list_users=admin__list_users,
    global_stats=admin__global_stats,
    export_global=admin__export_global,
    enter_broadcast=admin__enter_broadcast,
    handle_broadcast=admin__handle_broadcast,
)

inline_handler = SimpleNamespace(
    handle_inline=inline__handle_inline,
)

link_handler = SimpleNamespace(
    enter_link_mode=link__enter_link_mode,
    handle_input=link__handle_input,
    handle_message_id_input=link__handle_message_id_input,
)

mtproto_handler = SimpleNamespace(
    show_menu=mtproto__show_menu,
    start=mtproto__start,
    stop=mtproto__stop,
    restart=mtproto__restart,
    refresh_status=mtproto__refresh_status,
    enter_screenshot_mode=mtproto__enter_screenshot_mode,
    handle_screenshot_input=mtproto__handle_screenshot_input,
    enter_download_mode=mtproto__enter_download_mode,
    handle_download_input=mtproto__handle_download_input,
    handle_msgid_input=mtproto__handle_msgid_input,
)

media_browser_handler = SimpleNamespace(
    show_categories=mb__show_categories,
    show_category_items=mb__show_category_items,
    download_single=mb__download_single,
    download_all=mb__download_all,
    back_to_categories=mb__back_to_categories,
)

vc_handler = SimpleNamespace(
    show_menu=vc__show_menu,
    start_tour=vc__start_tour,
    pause_tour=vc__pause_tour,
    resume_tour=vc__resume_tour,
    stop_tour=vc__stop_tour,
    show_current=vc__show_current,
    show_history=vc__show_history,
    refresh_groups=vc__refresh_groups,
    refresh_status=vc__refresh_status,
    show_manual_menu=vc__show_manual_menu,
    enter_join_target=vc__enter_join_target,
    handle_join_target=vc__handle_join_target,
    leave_current=vc__leave_current,
    check_target=vc__check_target,
    handle_check_target=vc__handle_check_target,
    stay_5min=vc__stay_5min,
    enter_custom_stay=vc__enter_custom_stay,
    handle_custom_stay=vc__handle_custom_stay,
    show_settings=vc__show_settings,
)


# ===========================================================================
# Handlers — menu (central router + message router)
# ===========================================================================


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    is_admin = user.id in config.admin_ids
    await repo.upsert_user(user.id, user.username, user.first_name, is_admin)
    await repo.ensure_settings(user.id)
    await states.reset(user.id)

    text = msg.WELCOME if not update.callback_query else msg.MAIN_MENU_TEXT
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        reply_markup=kb.main_menu(),
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg.HELP_MENU_TEXT,
        reply_markup=kb.help_menu(),
        parse_mode="Markdown",
    )


async def _menu_edit(update: Update, text: str, markup: InlineKeyboardMarkup) -> None:
    q = update.callback_query
    try:
        await q.edit_message_text(text=text, reply_markup=markup, parse_mode="Markdown")
    except Exception as exc:  # noqa: BLE001
        logger.debug("edit_message_text failed: %s", exc)


async def _menu_answer(update: Update, text: str | None = None) -> None:
    try:
        await update.callback_query.answer(text)
    except Exception:  # noqa: BLE001
        pass


async def route_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if q is None or q.data is None:
        return
    data: str = q.data
    user = update.effective_user
    if user:
        is_admin = user.id in config.admin_ids
        await repo.upsert_user(user.id, user.username, user.first_name, is_admin)

    try:
        # ---- Menu navigation: m:<menu> ----
        if data == "m:main":
            await states.reset(user.id)
            await _menu_edit(update, msg.MAIN_MENU_TEXT, kb.main_menu())
        elif data == "m:dl":
            await states.reset(user.id)
            await _menu_edit(update, msg.DOWNLOAD_MENU_TEXT, kb.download_menu())
        elif data == "m:ai":
            await states.reset(user.id)
            await _menu_edit(update, msg.ANALYZE_MENU_TEXT, kb.analyze_menu())
        elif data == "m:ins":
            await states.reset(user.id)
            await _menu_edit(update, msg.INSPECTOR_MENU_TEXT, kb.inspector_menu())
        elif data == "m:tb":
            await states.reset(user.id)
            await _menu_edit(update, msg.TOOLBOX_MENU_TEXT, kb.toolbox_menu())
        elif data == "m:lib":
            await states.reset(user.id)
            await _menu_edit(update, msg.LIBRARY_MENU_TEXT, kb.library_menu())
        elif data == "m:st":
            await states.reset(user.id)
            await stats_handler.show_menu(update, context)
        elif data == "m:qr":
            await states.reset(user.id)
            await _menu_edit(update, msg.QR_MENU_TEXT, kb.qr_menu())
        elif data == "m:batch":
            await states.reset(user.id)
            await _menu_edit(update, msg.BATCH_MENU_TEXT, kb.batch_menu())
        elif data == "m:sched":
            await states.reset(user.id)
            await _menu_edit(update, msg.SCHEDULED_MENU_TEXT, kb.scheduled_menu())
        elif data == "m:bk":
            await states.reset(user.id)
            await _menu_edit(update, msg.BACKUP_MENU_TEXT, kb.backup_menu())
        elif data == "m:admin":
            await admin_handler.show_menu(update, context)
        elif data == "m:hist":
            await states.reset(user.id)
            await _menu_edit(update, msg.HISTORY_MENU_TEXT, kb.history_menu())
        elif data == "m:set":
            await states.reset(user.id)
            s = await repo.ensure_settings(user.id)
            await _menu_edit(update, msg.settings_text(s), kb.settings_menu())
        elif data == "m:help":
            await states.reset(user.id)
            await _menu_edit(update, msg.HELP_MENU_TEXT, kb.help_menu())

        # ---- Back buttons: b:<menu> ----
        elif data == "b:main":
            await states.reset(user.id)
            await _menu_edit(update, msg.MAIN_MENU_TEXT, kb.main_menu())
        elif data == "b:dl":
            await states.reset(user.id)
            await _menu_edit(update, msg.DOWNLOAD_MENU_TEXT, kb.download_menu())
        elif data == "b:ai":
            await states.reset(user.id)
            await _menu_edit(update, msg.ANALYZE_MENU_TEXT, kb.analyze_menu())
        elif data == "b:ins":
            await states.reset(user.id)
            await _menu_edit(update, msg.INSPECTOR_MENU_TEXT, kb.inspector_menu())
        elif data == "b:tb":
            await states.reset(user.id)
            await _menu_edit(update, msg.TOOLBOX_MENU_TEXT, kb.toolbox_menu())
        elif data == "b:lib":
            await states.reset(user.id)
            await _menu_edit(update, msg.LIBRARY_MENU_TEXT, kb.library_menu())
        elif data == "b:st":
            await states.reset(user.id)
            await _menu_edit(update, msg.STATS_MENU_TEXT, kb.stats_menu())
        elif data == "b:qr":
            await states.reset(user.id)
            await _menu_edit(update, msg.QR_MENU_TEXT, kb.qr_menu())
        elif data == "b:batch":
            await states.reset(user.id)
            await _menu_edit(update, msg.BATCH_MENU_TEXT, kb.batch_menu())
        elif data == "b:sched":
            await states.reset(user.id)
            await _menu_edit(update, msg.SCHEDULED_MENU_TEXT, kb.scheduled_menu())
        elif data == "b:bk":
            await states.reset(user.id)
            await _menu_edit(update, msg.BACKUP_MENU_TEXT, kb.backup_menu())
        elif data == "b:admin":
            await admin_handler.show_menu(update, context)
        elif data == "b:hist":
            await _menu_edit(update, msg.HISTORY_MENU_TEXT, kb.history_menu())
        elif data == "b:set":
            s = await repo.ensure_settings(user.id)
            await _menu_edit(update, msg.settings_text(s), kb.settings_menu())
        elif data == "b:help":
            await _menu_edit(update, msg.HELP_MENU_TEXT, kb.help_menu())

        # ---- Download flow ----
        elif data == "dl:fwd":
            await download_handler.enter_forward_mode(update, context)
        elif data == "dl:link":
            await link_handler.enter_link_mode(update, context)
        elif data == "dl:cancel":
            await states.reset(user.id)
            await _menu_edit(update, msg.DOWNLOAD_CANCEL_PROMPT, kb.download_menu())

        # ---- Media browser (channel category browsing + Download All) ----
        elif data.startswith("mb:cat:"):
            category = data.split(":", 2)[2]
            await media_browser_handler.show_category_items(update, context, category)
        elif data.startswith("mb:one:"):
            try:
                mid = int(data.split(":", 2)[2])
            except ValueError:
                await update.callback_query.answer("Invalid id.")
            else:
                await media_browser_handler.download_single(update, context, mid)
        elif data.startswith("mb:all:"):
            category = data.split(":", 2)[2]
            await media_browser_handler.download_all(update, context, category)
        elif data == "mb:back":
            await media_browser_handler.back_to_categories(update, context)

        # ---- Analyze flow ----
        elif data == "ai:fwd":
            await analyze_handler.enter_analyze_mode(update, context)
        elif data == "ai:modes":
            await analyze_handler.show_modes(update, context)
        elif data.startswith("ai:mode:"):
            mode = data.split(":", 2)[2]
            await analyze_handler.set_mode(update, context, mode)
        elif data == "ai:cancel":
            await states.reset(user.id)
            await _menu_edit(update, msg.ANALYZE_CANCEL_PROMPT, kb.analyze_menu())

        # ---- Inspector ----
        elif data == "ins:fwd":
            await inspector_handler.enter_inspect_mode(update, context)
        elif data == "ins:recent":
            await inspector_handler.show_recent(update, context)

        # ---- Toolbox ----
        elif data.startswith("tb:"):
            tool = data.split(":", 1)[1]
            if tool == "cancel":
                await states.reset(user.id)
                await _menu_edit(update, msg.TOOLBOX_MENU_TEXT, kb.toolbox_menu())
            else:
                await toolbox_handler.handle_tool_selection(update, context, tool)

        # ---- Library ----
        elif data.startswith("lib:"):
            await library_handler.handle_action(update, context, data)

        # ---- Stats ----
        elif data == "st:me":
            await stats_handler.show_user(update, context)
        elif data == "st:global":
            await stats_handler.show_global(update, context)

        # ---- QR ----
        elif data == "qr:make":
            await qr_handler.enter_make_mode(update, context)
        elif data == "qr:cancel":
            await states.reset(user.id)
            await _menu_edit(update, msg.QR_MENU_TEXT, kb.qr_menu())

        # ---- Batch / album ----
        elif data == "batch:fwd":
            await batch_handler.enter_batch_mode(update, context)

        # ---- Scheduled ----
        elif data == "sched:list":
            await scheduled_handler.show_list(update, context)

        # ---- Backup ----
        elif data == "bk:export":
            await backup_handler.do_export(update, context)
        elif data == "bk:import":
            await backup_handler.enter_import(update, context)

        # ---- Admin ----
        elif data == "adm:users":
            await admin_handler.list_users(update, context)
        elif data == "adm:stats":
            await admin_handler.global_stats(update, context)
        elif data == "adm:export":
            await admin_handler.export_global(update, context)
        elif data == "adm:bcast":
            await admin_handler.enter_broadcast(update, context)
        elif data == "adm:mtp":
            await mtproto_handler.show_menu(update, context)

        # ---- MTProto backend (admin) ----
        elif data == "mtp:start":
            await mtproto_handler.start(update, context)
        elif data == "mtp:stop":
            await mtproto_handler.stop(update, context)
        elif data == "mtp:restart":
            await mtproto_handler.restart(update, context)
        elif data == "mtp:status":
            await mtproto_handler.refresh_status(update, context)
        elif data == "mtp:screenshot":
            await mtproto_handler.enter_screenshot_mode(update, context)
        elif data == "mtp:download":
            await mtproto_handler.enter_download_mode(update, context)

        # ---- VC Tour (admin) ----
        elif data == "adm:vc":
            await vc_handler.show_menu(update, context)
        elif data == "vc:start":
            await vc_handler.start_tour(update, context)
        elif data == "vc:pause":
            await vc_handler.pause_tour(update, context)
        elif data == "vc:resume":
            await vc_handler.resume_tour(update, context)
        elif data == "vc:stop":
            await vc_handler.stop_tour(update, context)
        elif data == "vc:current":
            await vc_handler.show_current(update, context)
        elif data == "vc:status":
            await vc_handler.refresh_status(update, context)
        elif data.startswith("vc:hist:"):
            try:
                page = int(data.split(":")[2])
            except (ValueError, IndexError):
                page = 0
            await vc_handler.show_history(update, context, page)
        elif data == "vc:refresh":
            await vc_handler.refresh_groups(update, context)
        elif data == "vc:manual":
            await vc_handler.show_manual_menu(update, context)
        elif data == "vc:settings":
            await vc_handler.show_settings(update, context)
        elif data == "vc:jointarget":
            await vc_handler.enter_join_target(update, context)
        elif data == "vc:leave":
            await vc_handler.leave_current(update, context)
        elif data == "vc:stay5":
            await vc_handler.stay_5min(update, context)
        elif data == "vc:staycustom":
            await vc_handler.enter_custom_stay(update, context)
        elif data == "vc:checktarget":
            await vc_handler.check_target(update, context)
        elif data == "b:vc":
            await vc_handler.show_menu(update, context)
        elif data == "b:vcmanual":
            await vc_handler.show_manual_menu(update, context)

        # ---- History ----
        elif data == "h:dl":
            await history_handler.show_downloads(update, context)
        elif data == "h:ai":
            await history_handler.show_ai(update, context)
        elif data.startswith("h:clr"):
            await history_handler.clear(update, context, data)

        # ---- Settings cycling ----
        elif data.startswith("s:"):
            await settings_handler.handle_setting(update, context, data)

        # ---- Help sections ----
        elif data.startswith("hp:"):
            await help_handler.show_section(update, context, data)

        else:
            await _menu_answer(update, "Unknown action.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("route_callback error for data=%s: %s", data, exc)
        try:
            await q.answer("⚠️ Something went wrong. Please try again.")
        except Exception:  # noqa: BLE001
            pass


async def route_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return
    user_id = update.effective_user.id
    is_admin = user_id in config.admin_ids
    await repo.upsert_user(user_id, update.effective_user.username,
                           update.effective_user.first_name, is_admin)

    state = await states.get_state(user_id)
    tool = await states.get_tool(user_id)

    if state == states.AWAIT_DOWNLOAD_FORWARD:
        await download_handler.handle_forwarded_media(update, context)
    elif state == states.AWAIT_ANALYZE:
        await analyze_handler.handle_forwarded_video(update, context)
    elif state == states.AWAIT_INSPECT:
        await inspector_handler.handle_inspect_input(update, context)
    elif state == states.AWAIT_TOOLBOX:
        if tool == "imgconv" and update.message.text and not _has_media(update):
            await toolbox_handler.handle_imgconv_format(update, context)
        else:
            await toolbox_handler.handle_forwarded_media(update, context)
    elif state == states.AWAIT_LIBRARY_SEARCH:
        await library_handler.handle_search_input(update, context)
    elif state == states.AWAIT_QR:
        await qr_handler.handle_input(update, context)
    elif state == states.AWAIT_BATCH:
        await batch_handler.handle_batch_message(update, context)
    elif state == states.AWAIT_BACKUP_IMPORT:
        await backup_handler.handle_import_file(update, context)
    elif state == states.AWAIT_ADMIN_BCAST:
        await admin_handler.handle_broadcast(update, context)
    elif state == "await_gemini_key":
        await settings_handler.handle_gemini_key_input(update, context)
    elif state == states.AWAIT_SCHEDULE:
        await scheduled_handler.handle_schedule_input(update, context)
    elif state == states.AWAIT_LINK_DOWNLOAD:
        await link_handler.handle_input(update, context)
    elif state == states.AWAIT_LINK_MESSAGE_ID:
        await link_handler.handle_message_id_input(update, context)
    elif state == states.AWAIT_MTPROTO_SCREENSHOT:
        await mtproto_handler.handle_screenshot_input(update, context)
    elif state == states.AWAIT_MTPROTO_DOWNLOAD:
        await mtproto_handler.handle_download_input(update, context)
    elif state == states.AWAIT_MTPROTO_MSGID:
        await mtproto_handler.handle_msgid_input(update, context)
    elif state == states.AWAIT_VC_JOIN_TARGET:
        await vc_handler.handle_join_target(update, context)
    elif state == states.AWAIT_VC_CHECK_TARGET:
        await vc_handler.handle_check_target(update, context)
    elif state == states.AWAIT_VC_STAY:
        await vc_handler.handle_custom_stay(update, context)
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=msg.MAIN_MENU_TEXT,
            reply_markup=kb.main_menu(),
            parse_mode="Markdown",
        )


def _has_media(update: Update) -> bool:
    m = update.message
    return any(getattr(m, attr, None) for attr in (
        "video", "document", "audio", "photo", "voice", "video_note",
        "animation", "sticker",
    ))


# ===========================================================================
# Main entry point
# ===========================================================================


async def _on_error(update: object, context) -> None:
    """PTB global error handler.

    Any exception raised inside a handler that isn't caught locally ends up
    here. We log it with full context so it's debuggable, but the bot keeps
    running — a single bad update never terminates the process.
    """
    error = getattr(context, "error", None)
    if error is None:
        logger.error("PTB error handler invoked with no error object.")
        return
    # Extract the update id for traceability.
    update_id = getattr(update, "update_id", "?")
    logger.error(
        "Unhandled exception in update %s: %s: %s",
        update_id, type(error).__name__, error,
        exc_info=context.error,
    )
    # Best-effort: notify the user that something went wrong (only for
    # updates that carry a chat). We use plain text to avoid any chance of
    # a secondary parse error.
    try:
        if isinstance(error, (BadRequest, Forbidden)):
            # These are usually user-facing (chat not found, blocked, etc.)
            # — don't try to reply, just log.
            return
        chat = getattr(getattr(update, "effective_chat", None), "id", None)
        if chat and not isinstance(error, (BadRequest, Forbidden)):
            await context.bot.send_message(
                chat_id=chat,
                text=(
                    "⚠️ An unexpected error occurred while processing your "
                    "request. Please try again, or use /start to return to "
                    "the main menu."
                ),
            )
    except Exception:  # noqa: BLE001
        pass


async def _scheduled_loop(app: Application) -> None:
    """Periodically run due scheduled tasks."""
    logger.info("Scheduled-task loop started (interval=60s).")
    while True:
        try:
            await asyncio.sleep(60)
            n = await scheduled_handler.run_due_tasks(app)
            if n:
                logger.info("Processed %d scheduled tasks.", n)
        except asyncio.CancelledError:
            logger.info("Scheduled-task loop cancelled.")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("scheduled loop error: %s", exc)


async def _post_init(app: Application) -> None:
    await init_db()
    # Register the Bot instance so background services (MTProto capture
    # handler) can send messages without needing the Application object.
    register_bot(app.bot)
    app._bg_tasks = [  # type: ignore[attr-defined]
        asyncio.create_task(
            periodic_cleanup(
                config.downloads_dir, config.frames_dir,
                interval=600, max_age=3600,
            )
        ),
        asyncio.create_task(_scheduled_loop(app)),
    ]
    # Start the MTProto backend if configured (background service).
    if config.mtproto_configured:
        try:
            result = await mtproto_manager.start()
            logger.info("MTProto backend startup: %s", result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MTProto backend failed to start (non-fatal): %s", exc)
        # Register VC tour in-group command handler + reconcile stale state.
        try:
            await vc_tour.reconcile_on_startup()
            if config.vc_tour_enabled:
                await vc_tour.register_command_handler()
                logger.info("VC tour command handler registered.")
            else:
                logger.info("VC tour disabled (VC_TOUR_ENABLED=false).")
        except Exception as exc:  # noqa: BLE001
            logger.warning("VC tour init failed (non-fatal): %s", exc)
    else:
        logger.info("MTProto backend disabled (MTPROTO_ENABLED=false).")
    logger.info("Bot post-init complete (DB + cleanup + scheduled loop + MTProto + VC).")


async def _post_stop(app: Application) -> None:
    for task in getattr(app, "_bg_tasks", []):  # type: ignore[attr-defined]
        task.cancel()
    # Stop the MTProto backend.
    if config.mtproto_configured:
        # Stop the VC tour first (leave active VC cleanly).
        try:
            await vc_tour.stop_tour()
        except Exception as exc:  # noqa: BLE001
            logger.debug("VC tour stop error: %s", exc)
        try:
            await mtproto_manager.stop()
        except Exception as exc:  # noqa: BLE001
            logger.debug("MTProto stop error: %s", exc)
    await close_db()
    logger.info("Bot stopped cleanly.")


def build_application() -> Application:
    if not config.bot_token:
        logger.error("TG_BOT_TOKEN is not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    builder = (
        ApplicationBuilder()
        .token(config.bot_token)
        .concurrent_updates(True)
        .post_init(_post_init)
        .post_stop(_post_stop)
        .read_timeout(config.download_timeout)
        .write_timeout(config.download_timeout)
        .connect_timeout(60)
        .pool_timeout(60)
    )
    if config.local_mode:
        builder = builder.base_url(config.bot_api_base_url)
        builder = builder.base_file_url(config.bot_api_file_url)
        builder = builder.local_mode(True)
        logger.info("Local Bot API server mode ENABLED: %s", config.bot_api_base_url)
    app = builder.build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))

    if config.inline_enabled:
        app.add_handler(InlineQueryHandler(inline_handler.handle_inline))

    app.add_handler(CallbackQueryHandler(route_callback))

    app.add_handler(MessageHandler(
        filters.ALL & ~filters.COMMAND, route_message
    ))

    # Global error handler — catches any unhandled exception so a single bad
    # update never crashes the bot.
    app.add_error_handler(_on_error)

    return app


def main() -> None:
    logger.info("=== MediaGrab AI Bot starting (Bot API only) ===")
    logger.info(
        "Limits: download=%dMB, upload=%dMB, concurrent=%d, retries=%d, "
        "frames=%d, inline=%s, webhook=%s",
        config.max_file_size_mb,
        config.upload_limit_bytes // (1024 * 1024),
        config.max_concurrent_downloads, config.max_retries, config.num_frames,
        config.inline_enabled, config.use_webhook,
    )

    app = build_application()

    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: app.stop_running())
        except (NotImplementedError, RuntimeError):
            pass

    try:
        if config.use_webhook and config.webhook_url:
            logger.info("Starting in WEBHOOK mode at %s:%d%s",
                        config.webhook_listen, config.webhook_port,
                        config.webhook_path)
            app.run_webhook(
                listen=config.webhook_listen,
                port=config.webhook_port,
                url_path=config.webhook_path,
                webhook_url=config.webhook_url + config.webhook_path,
                allowed_updates=Update.ALL_TYPES,
            )
        else:
            app.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("=== MediaGrab AI Bot stopped ===")


if __name__ == "__main__":
    main()
