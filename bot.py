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

# 📄 PDF через Platypus
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
    Строгая структура: title, short_description, summary, key_tasks, action_plan, conclusion.
    Ничего не придумываем, только из исходного текста.
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
        logger.warning("Не удалось распарсить JSON, fallback")
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

    # Заголовок и описание — строки
    if not isinstance(data.get("title"), str):
        data["title"] = str(data.get("title", ""))[:120]
    if not isinstance(data.get("short_description"), str):
        data["short_description"] = str(data.get("short_description", ""))[:400]

    return lang, data


# ---------- Текстовые утилиты ----------

def _normalize_bullets_list(raw: List[str]) -> List[str]:
    """Чистим список пунктов: строки, убираем лишние пробелы/переводы."""
    cleaned: List[str] = []
    for item in raw:
        if not item:
            continue
        text = " ".join(str(item).split())
        if text:
            cleaned.append(text)
    return cleaned


# ---------- PDF: отдельные страницы + хедер/футер ----------

def build_pdf(lang: str, data: Dict) -> bytes:
    """
    Макет:
    - дата/время вверху слева на каждой странице;
    - снизу линия + под ней название бота;
    - 1-я страница: title по центру, ниже H2 (short_description), БЕЗ текста;
    - 2-я страница: Summary;
    - 3-я страница: Key tasks;
    - 4-я страница: Action plan;
    - 5-я страница: Conclusion;
    - авто-перенос строк, текст не вылезает за поля.
    """
    buf = io.BytesIO()
    width, height = A4

    created_at = datetime.now().strftime("%d.%m.%Y %H:%M")
    created_label = t(lang, "Создано: ", "Created: ") + created_at
    footer_text = "summarinotebot"

    def add_page_frame(canvas, doc):
        canvas.saveState()

        # Хедер — дата
        canvas.setFont(FONT_NAME, 9)
        canvas.drawString(doc.leftMargin, height - 30, created_label)

        # Линия над футером
        line_y = 35
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, line_y, width - doc.rightMargin, line_y)

        # Футер — название бота под линией
        footer_y = 22
        fw = canvas.stringWidth(footer_text, FONT_NAME, 9)
        canvas.setFont(FONT_NAME, 9)
        canvas.drawString((width - fw) / 2, footer_y, footer_text)

        canvas.restoreState()

    # Чуть шире поля, чтобы текст не казался «сжатым»
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=32,
        rightMargin=32,
        topMargin=70,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    # Базовый стиль
    base = styles["Normal"]
    base.fontName = FONT_NAME
    base.fontSize = 11
    base.leading = 15

    # TITLE (по центру)
    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=24,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=10,
    )

    # H2 под title — краткое описание
    short_style = ParagraphStyle(
        "ShortDesc",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        spaceAfter=0,
    )

    # Заголовки секций
    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=16,
        leading=20,
        alignment=TA_LEFT,
        spaceBefore=10,
        spaceAfter=8,
    )

    # Текст списков — чуть «шире» (меньше отступов)
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=11,
        leading=15,
        leftIndent=0,
        spaceAfter=2,
    )

    story: List = []

    title = data.get("title") or t(lang, "Конспект", "Summary")
    short = data.get("short_description") or ""

    # ---------- 1-я страница: только title + H2 по центру ----------
    # Подняли/опустили, чтобы визуально было ближе к середине листа
    story.append(Spacer(1, height * 0.25))  # регулирует «по середине»
    story.append(Paragraph(title, title_style))
    if short:
        story.append(Spacer(1, 8))
        story.append(Paragraph(short, short_style))

    # Никакого текста на первой странице → сразу разрыв
    story.append(PageBreak())

    def section_elements(heading: str, bullets: List[str]) -> List:
        bullets_norm = _normalize_bullets_list(bullets)
        if not bullets_norm:
            return []

        elements: List = []
        elements.append(Paragraph(heading, heading_style))

        items = []
        for b in bullets_norm:
            p = Paragraph(b, bullet_style)
            items.append(ListItem(p, leftIndent=6))

        elements.append(
            ListFlowable(
                items,
                bulletType="bullet",
                bulletFontName=FONT_NAME,
                bulletFontSize=11,
                bulletIndent=0,
                leftIndent=14,
                spaceBefore=4,
                spaceAfter=6,
            )
        )
        return elements

    # ---------- 2-я страница: Summary ----------
    story.extend(section_elements(
        t(lang, "Краткое содержание", "Summary"),
        data.get("summary") or [],
    ))

    # ---------- 3-я страница: Key tasks ----------
    story.append(PageBreak())
    story.extend(section_elements(
        t(lang, "Ключевые задачи", "Key tasks"),
        data.get("key_tasks") or [],
    ))

    # ---------- 4-я страница: Action plan ----------
    story.append(PageBreak())
    story.extend(section_elements(
        t(lang, "План действий", "Action plan"),
        data.get("action_plan") or [],
    ))

    # ---------- 5-я страница: Conclusion ----------
    story.append(PageBreak())
    story.extend(section_elements(
        t(lang, "Итог", "Conclusion"),
        data.get("conclusion") or [],
    ))

    doc.build(story, onFirstPage=add_page_frame, onLaterPages=add_page_frame)

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

    context.chat_data["last_lang"] = lang
    context.chat_data["last_structured"] = data

    keyboard = [[InlineKeyboardButton("📄 PDF", callback_data="format_pdf")]]

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

