import logging
import os
import io
import json
import tempfile
import subprocess
from typing import Dict, List, Tuple
from datetime import datetime

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 📄 Нормальный PDF через Platypus (авто-перенос и новые страницы)
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER


# ---------- PDF-шрифт ----------

FONT_NAME = "DejaVuSans"  # файл DejaVuSans.ttf должен лежать рядом с bot.py
pdfmetrics.registerFont(TTFont(FONT_NAME, "DejaVuSans.ttf"))


# ---------- Конфиг ----------

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_AUDIO_BYTES = 24 * 1024 * 1024  # ~24MB

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


# ---------- Утилиты ----------

def detect_language(text: str) -> str:
    """Примитивно: если есть кириллица — ru, иначе en."""
    for ch in text:
        if "а" <= ch.lower() <= "я" or ch in "ёЁ":
            return "ru"
    return "en"


def t(lang: str, ru: str, en: str) -> str:
    return ru if lang == "ru" else en


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
            model="gpt-4o-mini-transcribe",  # можно заменить на "whisper-1"
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

    # Нормализация списков
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


# ---------- Вспомогательные функции для текста ----------

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
        text = " ".join(str(item).split())
        if text:
            cleaned.append(text)
    return cleaned


# ---------- PDF (автоперенос и новые страницы) ----------

def build_pdf(lang: str, data: Dict) -> bytes:
    """
    Аккуратный PDF:
    - нормальные отступы
    - автоперенос строк
    - автоматическое добавление новых страниц, если текста много
    - списки через bullets
    """
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=60,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Базовый стиль
    base = styles["Normal"]
    base.fontName = FONT_NAME
    base.fontSize = 11
    base.leading = 14

    # Заголовок (первая страница)
    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=16,
    )

    # Краткое описание
    short_style = ParagraphStyle(
        "ShortDesc",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    # Дата
    date_style = ParagraphStyle(
        "Date",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=9,
        leading=11,
        alignment=TA_LEFT,
        spaceAfter=15,
    )

    # Заголовки секций
    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=16,
        leading=20,
        alignment=TA_LEFT,
        spaceBefore=12,
        spaceAfter=8,
    )

    # Текст списков
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=11,
        leading=14,
        leftIndent=0,
    )

    story: List = []

    title = data.get("title") or t(lang, "Конспект", "Summary")
    short = data.get("short_description") or ""
    created_at = datetime.now().strftime("%d.%m.%Y %H:%M")
    created_label = t(lang, "Создано: ", "Created: ") + created_at

    # ---------- титульная часть ----------
    story.append(Paragraph(title, title_style))
    if short:
        story.append(Paragraph(short, short_style))
    story.append(Paragraph(created_label, date_style))
    story.append(Spacer(1, 12))

    # Можно явно добавить разрыв страницы после титула, если нужно
    story.append(PageBreak())

    # ---------- секции ----------
    def add_section(heading: str, bullets: List[str]):
        bullets_norm = _normalize_bullets_list(bullets)
        if not bullets_norm:
            return

        story.append(Paragraph(heading, heading_style))

        items = []
        for b in bullets_norm:
            p = Paragraph(b, bullet_style)
            items.append(ListItem(p, leftIndent=10))

        story.append(
            ListFlowable(
                items,
                bulletType="bullet",
                bulletFontName=FONT_NAME,
                bulletFontSize=11,
                bulletIndent=0,
                leftIndent=15,
                spaceBefore=4,
                spaceAfter=10,
            )
        )

    add_section(t(lang, "Краткое содержание", "Summary"), data.get("summary") or [])
    add_section(t(lang, "Ключевые задачи", "Key tasks"), data.get("key_tasks") or [])
    add_section(t(lang, "План действий", "Action plan"), data.get("action_plan") or [])
    add_section(t(lang, "Итог", "Conclusion"), data.get("conclusion") or [])

    # Platypus сам разобьёт story на страницы по высоте
    doc.build(story)

    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


# ---------- Telegram-хендлеры ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Привет! Отправьте голосовое или аудио, "
        "я сделаю аккуратную структурированную сводку и создам PDF.\n\n"
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

    status_msg = await message.reply_text("🔍 Анализирую аудио…")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input")
            output_path = os.path.join(tmpdir, "out.mp3")

            await tg_file.download_to_drive(input_path)
            ffmpeg_convert_to_mp3(input_path, output_path)

            raw_text = await transcribe_audio(output_path)
    except Exception:
        logger.exception("Ошибка на этапе аудио/ffmpeg/Whisper")
        await status_msg.edit_text(
            "Не смог распознать аудио 😔 Попробуйте ещё раз, лучше в формате OGG/MP3."
        )
        return

    if not raw_text or not raw_text.strip():
        await status_msg.edit_text("Я ничего не услышал в этом аудио 😔")
        return

    try:
        lang, data = await structure_text(raw_text)
    except Exception:
        logger.exception("Ошибка при структурировании текста")
        lang = detect_language(raw_text)
        data = {
            "title": raw_text[:80],
            "short_description": raw_text[:200],
            "summary": [raw_text[:1000]],
            "key_tasks": [],
            "action_plan": [],
            "conclusion": [],
        }

    # сохраним в chat_data
    context.chat_data["last_lang"] = lang
    context.chat_data["last_structured"] = data

    keyboard = [
        [InlineKeyboardButton("📄 PDF", callback_data="format_pdf")]
    ]

    text = t(
        lang,
        "✨ Финальный штрих…\n\nСгенерировать PDF-конспект?",
        "✨ Final touch…\n\nGenerate PDF summary?",
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


async def send_pdf(query, data: Dict, lang: str):
    await query.answer(
        t(lang, "Создаю PDF…", "Creating PDF…"),
        show_alert=False,
    )
    try:
        pdf_bytes = build_pdf(lang, data)
    except Exception:
        logger.exception("Ошибка при генерации PDF")
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
    main()
