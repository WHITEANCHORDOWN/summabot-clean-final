import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
import os
import io
import json
import tempfile
import subprocess
from typing import Dict, List, Tuple

from datetime import datetime
from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
# Unicode-шрифт для PDF (поддерживает кириллицу)
FONT_NAME = "DejaVuSans"
pdfmetrics.registerFont(TTFont(FONT_NAME, "DejaVuSans.ttf"))


# Google API (для Slides; если не установлено/не настроено – просто не будет работать этот формат)
try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
except ImportError:
    Credentials = None
    build = None


# ---------- Конфиг ----------

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")  # содержимое JSON серв. аккаунта

MAX_AUDIO_BYTES = 24 * 1024 * 1024  # ~24MB лимит

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Нет TELEGRAM_BOT_TOKEN")
if not OPENAI_API_KEY:
    raise RuntimeError("Нет OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- Мини HTTP-сервер для Render (healthcheck) ----------

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        # Глушим лишний шум в логах
        return


def start_health_server():
    """Простой HTTP-сервер, чтобы Render видел открытый порт."""
    port = int(os.environ.get("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Health server listening on port {port}")
    server.serve_forever()


_SLIDES_SERVICE = None
_DRIVE_SERVICE = None


# ---------- Утилиты ----------

def detect_language(text: str) -> str:
    """Примитивная проверка: если есть кириллица — ru, иначе en."""
    for ch in text:
        if "а" <= ch.lower() <= "я" or ch in "ёЁ":
            return "ru"
    return "en"


def t(lang: str, ru: str, en: str) -> str:
    return ru if lang == "ru" else en


def ensure_google_services():
    """Создаём клиенты Google Slides/Drive из сервисного аккаунта."""
    global _SLIDES_SERVICE, _DRIVE_SERVICE
    if _SLIDES_SERVICE and _DRIVE_SERVICE:
        return _SLIDES_SERVICE, _DRIVE_SERVICE

    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON не задан")

    if Credentials is None or build is None:
        raise RuntimeError("Нет библиотек google-api-python-client/google-auth")

    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    scopes = [
        "https://www.googleapis.com/auth/presentations",
        "https://www.googleapis.com/auth/drive.file",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    _SLIDES_SERVICE = build("slides", "v1", credentials=creds)
    _DRIVE_SERVICE = build("drive", "v3", credentials=creds)
    return _SLIDES_SERVICE, _DRIVE_SERVICE


def ffmpeg_convert_to_mp3(input_path: str, output_path: str) -> None:
    """Конвертация любого аудио в mp3 mono 16kHz."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            output_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------- OpenAI ----------

async def transcribe_audio(path: str) -> str:
    """Распознаём аудио в текст."""
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",  # можно заменить на whisper-1
            file=f,
            response_format="text",
        )
    return result


async def structure_text(raw_text: str) -> Tuple[str, Dict]:
    """
    Делаем строгую структуру: title, short_description, summary, key_tasks, action_plan, conclusion.
    Ничего не придумываем, только на основе текста.
    """
    lang = detect_language(raw_text)

    system_prompt = (
        "You are a strict summarizer. You ONLY use information from the user's text. "
        "You never invent facts, names or tasks that are not explicitly present. "
        "Respond strictly as JSON with keys: "
        "title, short_description, summary, key_tasks, action_plan, conclusion. "
        "Lists must be concise bullet points (3–10 items)."
    )

    if lang == "ru":
        user_prompt = (
            "Сделай структурированный конспект текста ниже БЕЗ воды и без выдумки. "
            "Не добавляй ничего, чего нет в тексте. "
            "Верни ОТВЕТ СТРОГО в JSON с ключами: "
            "title, short_description, summary, key_tasks, action_plan, conclusion.\n\n"
            f"Текст:\n{raw_text}"
        )
    else:
        user_prompt = (
            "Create a structured, concise summary of the text below with NO fluff and no invention. "
            "Return STRICT JSON with keys: title, short_description, summary, key_tasks, action_plan, conclusion.\n\n"
            f"Text:\n{raw_text}"
        )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = completion.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Не удалось распарсить JSON, возвращаю fallback")
        data = {
            "title": raw_text[:80],
            "short_description": raw_text[:200],
            "summary": [raw_text[:1000]],
            "key_tasks": [],
            "action_plan": [],
            "conclusion": [],
        }
        return lang, data

    # Нормализация: всё, что должно быть списками — превращаем в списки строк
    for key in ["summary", "key_tasks", "action_plan", "conclusion"]:
        value = data.get(key)
        if isinstance(value, str):
            data[key] = [value]
        elif isinstance(value, list):
            data[key] = [str(x) for x in value if x]
        else:
            data[key] = []

    # Заголовки и описания — в строки
    if not isinstance(data.get("title"), str):
        data["title"] = str(data.get("title", ""))[:120]
    if not isinstance(data.get("short_description"), str):
        data["short_description"] = str(data.get("short_description", ""))[:400]

    return lang, data
def _normalize_bullets_list(raw: List[str]) -> List[str]:
    """
    Чистим список пунктов:
    - конвертируем в строки
    - убираем лишние переводы строк и двойные пробелы
    """
    cleaned: List[str] = []
    for item in raw:
        if not item:
            continue
        text = " ".join(str(item).split())  # все виды пробелов/переносов -> один пробел
        if text:
            cleaned.append(text)
    return cleaned


# ---------- PDF ----------

def _wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines = []
    line = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if line else 0)
        if cur_len + add > max_chars:
            lines.append(" ".join(line))
            line = [w]
            cur_len = len(w)
        else:
            line.append(w)
            cur_len += add
    if line:
        lines.append(" ".join(line))
    return lines or [""]

def build_pdf(lang: str, data: Dict) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    margin = 72  # ~2 см слева/справа
    title_font = 22
    heading_font = 16
    body_font = 11
    max_chars = 60  # чтобы строки точно не вылезали за край

    title = data.get("title") or t(lang, "Конспект", "Summary")
    short = data.get("short_description") or ""
    created_at = datetime.now().strftime("%d.%m.%Y %H:%M")

    # Положения хедера/футера
    header_text_y = height - 40
    header_line_y = header_text_y - 4
    footer_text_y = 40
    footer_line_y = footer_text_y + 6
    bottom_limit = footer_line_y + 25  # ниже этого не пишем текст

    date_text = t(lang, f"Создано: {created_at}", f"Created: {created_at}")

    # ---------- хедер и футер ----------

    def draw_header():
        """Дата/время сверху слева + тонкая линия."""
        c.setFont(FONT_NAME, 9)
        c.drawString(margin, header_text_y, date_text)
        c.setLineWidth(0.5)
        c.line(margin, header_line_y, width - margin, header_line_y)

    def draw_footer():
        """Имя бота снизу по центру + тонкая линия."""
        footer_text = "summarinotebot"
        footer_font = 9
        c.setLineWidth(0.5)
        c.line(margin, footer_line_y, width - margin, footer_line_y)
        c.setFont(FONT_NAME, footer_font)
        fw = c.stringWidth(footer_text, FONT_NAME, footer_font)
        c.drawString((width - fw) / 2, footer_text_y, footer_text)

    # ---------- титульная страница ----------

    draw_header()

    # Заголовок ближе к центру страницы
    c.setFont(FONT_NAME, title_font)
    title_w = c.stringWidth(title, FONT_NAME, title_font)
    c.drawString((width - title_w) / 2, height - 120, title)

    # Краткое описание под заголовком
    if short:
        c.setFont(FONT_NAME, body_font)
        text = c.beginText(margin, height - 170)
        for line in _wrap_text(short, max_chars):
            text.textLine(line)
        c.drawText(text)

    draw_footer()
    c.showPage()

    # ---------- вспомогательная функция для секций ----------

    def draw_section(heading: str, bullets: List[str]):
        bullets = _normalize_bullets_list(bullets)
        if not bullets:
            return

        # Новая страница секции
        draw_header()
        c.setFont(FONT_NAME, heading_font)
        c.drawString(margin, height - margin, heading)

        text = c.beginText(margin, height - margin - 30)
        text.setFont(FONT_NAME, body_font)

        for bullet in bullets:
            lines = _wrap_text(bullet, max_chars)
            for i, line in enumerate(lines):
                prefix = "• " if i == 0 else "   "
                text.textLine(prefix + line)

                # Если подходим к низу страницы — перенос
                if text.getY() < bottom_limit:
                    c.drawText(text)
                    draw_footer()
                    c.showPage()

                    # Новая страница с тем же заголовком
                    draw_header()
                    c.setFont(FONT_NAME, heading_font)
                    c.drawString(margin, height - margin, heading)
                    text = c.beginText(margin, height - margin - 30)
                    text.setFont(FONT_NAME, body_font)

            text.textLine("")  # пустая строка между пунктами

        c.drawText(text)
        draw_footer()
        c.showPage()

    # ---------- сами секции ----------

    draw_section(t(lang, "Краткое содержание", "Summary"), data.get("summary") or [])
    draw_section(t(lang, "Ключевые задачи", "Key tasks"), data.get("key_tasks") or [])
    draw_section(t(lang, "План действий", "Action plan"), data.get("action_plan") or [])
    draw_section(t(lang, "Итог", "Conclusion"), data.get("conclusion") or [])

    c.save()
    buf.seek(0)
    return buf.read()

    # ---------- сами секции ----------

    draw_section(t(lang, "Краткое содержание", "Summary"), data.get("summary") or [])
    draw_section(t(lang, "Ключевые задачи", "Key tasks"), data.get("key_tasks") or [])
    draw_section(t(lang, "План действий", "Action plan"), data.get("action_plan") or [])
    draw_section(t(lang, "Итог", "Conclusion"), data.get("conclusion") or [])

    c.save()
    buf.seek(0)
    return buf.read()

    # ---------- сами секции ----------
    draw_section(t(lang, "Краткое содержание", "Summary"), data.get("summary") or [])
    draw_section(t(lang, "Ключевые задачи", "Key tasks"), data.get("key_tasks") or [])
    draw_section(t(lang, "План действий", "Action plan"), data.get("action_plan") or [])
    draw_section(t(lang, "Итог", "Conclusion"), data.get("conclusion") or [])

    c.save()
    buf.seek(0)
    return buf.read()



    # ---------- титульная страница ----------
    c.setFont(FONT_NAME, 22)
    c.drawString(margin, height - margin - 10, title)

    c.setFont(FONT_NAME, 10)
    c.drawString(
        margin,
        height - margin - 35,
        t(lang, f"Создано: {created_at}", f"Created: {created_at}"),
    )

    if short:
        c.setFont(FONT_NAME, 11)
        text = c.beginText(margin, height - margin - 70)
        for line in _wrap_text(short, 90):
            text.textLine(line)
        c.drawText(text)

    c.showPage()

    # ---------- вспомогательная функция для секций ----------
    def draw_section(heading: str, bullets: List[str]):
        if not bullets:
            return

        nonlocal c
        c.setFont(FONT_NAME, 16)
        c.drawString(margin, height - margin, heading)

        text = c.beginText(margin, height - margin - 30)
        text.setFont(FONT_NAME, 11)

        for bullet in bullets:
            lines = _wrap_text(bullet, 90)
            for i, line in enumerate(lines):
                prefix = "• " if i == 0 else "   "
                text.textLine(prefix + line)

                # если подошли к низу страницы — перенос
                if text.getY() < margin + 40:
                    c.drawText(text)
                    c.showPage()
                    c.setFont(FONT_NAME, 16)
                    c.drawString(margin, height - margin, heading)
                    text = c.beginText(margin, height - margin - 30)
                    text.setFont(FONT_NAME, 11)

            text.textLine("")  # пустая строка между буллетами

        c.drawText(text)
        c.showPage()

    # ---------- сами секции ----------
    draw_section(t(lang, "Краткое содержание", "Summary"), data.get("summary") or [])
    draw_section(
        t(lang, "Ключевые задачи", "Key tasks"), data.get("key_tasks") or []
    )
    draw_section(
        t(lang, "План действий", "Action plan"), data.get("action_plan") or []
    )
    draw_section(t(lang, "Итог", "Conclusion"), data.get("conclusion") or [])

    c.save()
    buf.seek(0)
    return buf.read()

# ---------- Google Slides ----------

def _slides_title_and_bullets_requests(title: str, subtitle: str, slides_data: Dict[str, List[str]], lang: str):
    """Формируем batchUpdate запросы: титульный + 4 секции."""
    requests = []

    # Удалим дефолтный слайд в презентации позже, здесь только создаём свои.

    def title_slide():
        slide_id = "title-slide"
        title_shape_id = "title-box"
        subtitle_shape_id = "subtitle-box"
        return [
            {
                "createSlide": {
                    "objectId": slide_id,
                    "slideLayoutReference": {"predefinedLayout": "BLANK"},
                }
            },
            {
                "createShape": {
                    "objectId": title_shape_id,
                    "shapeType": "TEXT_BOX",
                    "elementProperties": {
                        "pageObjectId": slide_id,
                        "size": {
                            "width": {"magnitude": 8000000, "unit": "EMU"},
                            "height": {"magnitude": 800000, "unit": "EMU"},
                        },
                        "transform": {
                            "scaleX": 1,
                            "scaleY": 1,
                            "translateX": 800000,
                            "translateY": 800000,
                            "unit": "EMU",
                        },
                    },
                }
            },
            {
                "insertText": {
                    "objectId": title_shape_id,
                    "insertionIndex": 0,
                    "text": title,
                }
            },
            {
                "createShape": {
                    "objectId": subtitle_shape_id,
                    "shapeType": "TEXT_BOX",
                    "elementProperties": {
                        "pageObjectId": slide_id,
                        "size": {
                            "width": {"magnitude": 8000000, "unit": "EMU"},
                            "height": {"magnitude": 2000000, "unit": "EMU"},
                        },
                        "transform": {
                            "scaleX": 1,
                            "scaleY": 1,
                            "translateX": 800000,
                            "translateY": 2000000,
                            "unit": "EMU",
                        },
                    },
                }
            },
            {
                "insertText": {
                    "objectId": subtitle_shape_id,
                    "insertionIndex": 0,
                    "text": subtitle,
                }
            },
        ]

    def bullets_slides_for_section(title_text: str, bullets: List[str], base_id: str):
        reqs = []
        chunk_size = 7
        for idx in range(0, len(bullets), chunk_size):
            chunk = bullets[idx : idx + chunk_size]
            slide_id = f"{base_id}-{idx//chunk_size}"
            title_shape_id = f"title-{base_id}-{idx//chunk_size}"
            body_shape_id = f"body-{base_id}-{idx//chunk_size}"

            title_with_suffix = title_text if idx == 0 else f"{title_text} ({idx//chunk_size + 1})"

            reqs.extend(
                [
                    {
                        "createSlide": {
                            "objectId": slide_id,
                            "slideLayoutReference": {"predefinedLayout": "BLANK"},
                        }
                    },
                    {
                        "createShape": {
                            "objectId": title_shape_id,
                            "shapeType": "TEXT_BOX",
                            "elementProperties": {
                                "pageObjectId": slide_id,
                                "size": {
                                    "width": {"magnitude": 8000000, "unit": "EMU"},
                                    "height": {"magnitude": 800000, "unit": "EMU"},
                                },
                                "transform": {
                                    "scaleX": 1,
                                    "scaleY": 1,
                                    "translateX": 800000,
                                    "translateY": 600000,
                                    "unit": "EMU",
                                },
                            },
                        }
                    },
                    {
                        "insertText": {
                            "objectId": title_shape_id,
                            "insertionIndex": 0,
                            "text": title_with_suffix,
                        }
                    },
                    {
                        "createShape": {
                            "objectId": body_shape_id,
                            "shapeType": "TEXT_BOX",
                            "elementProperties": {
                                "pageObjectId": slide_id,
                                "size": {
                                    "width": {"magnitude": 8000000, "unit": "EMU"},
                                    "height": {"magnitude": 4000000, "unit": "EMU"},
                                },
                                "transform": {
                                    "scaleX": 1,
                                    "scaleY": 1,
                                    "translateX": 800000,
                                    "translateY": 1500000,
                                    "unit": "EMU",
                                },
                            },
                        },
                    },
                    {
                        "insertText": {
                            "objectId": body_shape_id,
                            "insertionIndex": 0,
                            "text": "\n".join(f"• {b}" for b in chunk),
                        }
                    },
                ]
            )
        return reqs

    requests.extend(title_slide())

    section_titles = {
        "summary": t(lang, "Краткое содержание", "Summary"),
        "key_tasks": t(lang, "Ключевые задачи", "Key tasks"),
        "action_plan": t(lang, "План действий", "Action plan"),
        "conclusion": t(lang, "Итог", "Conclusion"),
    }

    for key, bullets in slides_data.items():
        if not bullets:
            continue
        reqs = bullets_slides_for_section(section_titles[key], bullets, key)
        requests.extend(reqs)

    return requests


def build_slides(lang: str, data: Dict) -> str:
    slides_service, drive_service = ensure_google_services()

    title = data.get("title") or t(lang, "Конспект", "Summary")
    short = data.get("short_description") or ""

    presentation = slides_service.presentations().create(body={"title": title}).execute()
    pres_id = presentation["presentationId"]
    first_slide_id = presentation["slides"][0]["objectId"]

    slides_data = {
        "summary": data.get("summary") or [],
        "key_tasks": data.get("key_tasks") or [],
        "action_plan": data.get("action_plan") or [],
        "conclusion": data.get("conclusion") or [],
    }

    requests = [{"deleteObject": {"objectId": first_slide_id}}]
    requests += _slides_title_and_bullets_requests(title, short, slides_data, lang)

    slides_service.presentations().batchUpdate(
        presentationId=pres_id, body={"requests": requests}
    ).execute()

    # Делаем доступ по ссылке
    drive_service.permissions().create(
        fileId=pres_id,
        body={"role": "reader", "type": "anyone"},
    ).execute()

    return f"https://docs.google.com/presentation/d/{pres_id}/edit"


# ---------- Telegram-хендлеры ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Привет! Отправьте голосовое или аудио, "
        "я сделаю аккуратную сводку и предложу варианты скачивания.\n\n"
        "Поддерживаю русский и английский языки 🎧"
    )
    await update.message.reply_text(text)


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return

    tg_file = None
    file_size = None

    if message.voice:
        tg_file = await message.voice.get_file()
        file_size = message.voice.file_size
    elif message.audio:
        tg_file = await message.audio.get_file()
        file_size = message.audio.file_size
    elif (
        message.document
        and message.document.mime_type
        and message.document.mime_type.startswith("audio/")
    ):
        tg_file = await message.document.get_file()
        file_size = message.document.file_size
    else:
        await message.reply_text("Пока я работаю только с голосовыми и аудио-файлами 🎧")
        return

    if file_size and file_size > MAX_AUDIO_BYTES:
        await message.reply_text(
            "Файл слишком большой для распознавания (лимит ~24MB).\n"
            "Пожалуйста, отправьте более короткий фрагмент."
        )
        return

    # 1) Сообщение «анализирую»
    status_msg = await message.reply_text("🔍 Анализирую аудио…")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input")
            output_path = os.path.join(tmpdir, "out.mp3")

            await tg_file.download_to_drive(input_path)
            ffmpeg_convert_to_mp3(input_path, output_path)

            raw_text = await transcribe_audio(output_path)
    except Exception as e:
        logger.exception("Ошибка на этапе аудио/ffmpeg/Whisper: %s", e)
        await status_msg.edit_text(
            "Не смог распознать аудио 😔 Попробуйте ещё раз, лучше в формате OGG/MP3."
        )
        return

    if not raw_text or not raw_text.strip():
        await status_msg.edit_text("Я ничего не услышал в этом аудио 😔")
        return

    try:
        lang, data = await structure_text(raw_text)
    except Exception as e:
        logger.exception("Ошибка при структурировании текста: %s", e)
        lang = detect_language(raw_text)
        data = {
            "title": raw_text[:80],
            "short_description": raw_text[:200],
            "summary": [raw_text[:1000]],
            "key_tasks": [],
            "action_plan": [],
            "conclusion": [],
        }

    # Сохраняем в chat_data, чтобы использовать после выбора формата
    context.chat_data["last_lang"] = lang
    context.chat_data["last_structured"] = data

    # 2) «Финальный штрих» + выбор формата
    keyboard = [
        [
            InlineKeyboardButton("📄 PDF", callback_data="format_pdf"),
            InlineKeyboardButton("📊 Google Slides", callback_data="format_slides"),
        ]
    ]
    text = t(
        lang,
        "✨ Финальный штрих…\n\nВ каком формате хотите файл?",
        "✨ Final touch…\n\nWhich format do you want?",
    )

    await status_msg.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


async def handle_format_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = context.chat_data.get("last_structured")
    lang = context.chat_data.get("last_lang", "en")

    if not data:
        await query.edit_message_text(
            t(
                lang,
                "У меня нет свежего конспекта. Сначала отправьте голосовое или аудио.",
                "I don't see a recent transcript. Please send an audio message first.",
            )
        )
        return

    if query.data == "format_pdf":
        await send_pdf(query, data, lang)
    elif query.data == "format_slides":
        await send_slides(query, data, lang)


async def send_slides(query, data: Dict, lang: str):
    await query.answer(
        t(lang, "Создаю презентацию…", "Creating Google Slides deck…"),
        show_alert=False,
    )
    try:
        link = build_slides(lang, data)
    except Exception as e:
        logger.exception("Ошибка при генерации Slides: %s", e)
        await query.message.reply_text(
            t(
                lang,
                "Не удалось создать презентацию. Проверьте настройки Google API.",
                "Failed to create presentation. Please check Google API settings.",
            )
        )
        return

    await query.message.reply_text(
        t(
            lang,
            f"Готово! Вот ссылка на презентацию:\n{link}",
            f"Done! Here is your deck:\n{link}",
        )
    )

    # Предложим ещё формат
    keyboard = [
        [
            InlineKeyboardButton("📄 PDF", callback_data="format_pdf"),
            InlineKeyboardButton("📊 Google Slides", callback_data="format_slides"),
        ]
    ]
    await query.message.reply_text(
        t(
            lang,
            "Хотите также сохранить в другом формате?",
            "Do you also want another format?",
        ),
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
async def send_pdf(query, data: Dict, lang: str):
    # Показываем маленькое уведомление, клавиатура с кнопками остаётся
    await query.answer(
        t(lang, "Создаю PDF…", "Creating PDF…"),
        show_alert=False,
    )
    try:
        pdf_bytes = build_pdf(lang, data)
    except Exception as e:
        logger.exception("Ошибка при генерации PDF: %s", e)
        await query.message.reply_text(
            t(
                lang,
                "Не удалось создать PDF. Попробуйте позже.",
                "Failed to create PDF. Please try again later.",
            )
        )
        return

    filename = (data.get("title") or "summary").replace(" ", "_")[:50] + ".pdf"
    await query.message.reply_document(
        document=pdf_bytes,
        filename=filename,
        caption=t(lang, "Вот ваш PDF-конспект 🤓", "Here is your PDF summary 🤓"),
    )


# ---------- main ----------

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(
            filters.VOICE | filters.AUDIO | filters.Document.AUDIO,
            handle_audio,
        )
    )
    app.add_handler(CallbackQueryHandler(handle_format_choice))

    logger.info("Bot started (polling)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    # 1) Запускаем мини веб-сервер в фоне (для Render)
    threading.Thread(target=start_health_server, daemon=True).start()

    # 2) Запускаем Telegram-бота (polling)
    main()
