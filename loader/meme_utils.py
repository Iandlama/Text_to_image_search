import re

# Список сообщений из memecap: [{"content": ..., "role": ...}, ...]
Messages = list[dict[str, str]]

# --- Парсинг VQA-промпта memecap в чистые поля для visual IR ---
# Исходный user-content выглядит как:
#   This is a meme with the title "<title>". The image description is
#   "<image_desc>".What is the meme poster trying to convey? Answer:
# Это промпт для LLM, а не документ для поиска — эмбеддить его целиком нельзя
# (боилерплейт повторяется во всех записях и становится шумом). Разбираем на поля.

TITLE_RE = re.compile(r'the title "(.*?)"\. The image description is ', re.DOTALL)
DESC_RE = re.compile(
    r'The image description is "(.*)"\.\s*What is the meme poster trying to convey',
    re.DOTALL,
)


def norm(s: str | None) -> str:
    """Схлопывает переносы строк и повторные пробелы в одну строку."""
    return re.sub(r'\s+', ' ', (s or '').replace('\n', ' ')).strip()


def parse_caption(
    messages: Messages | str | None,
) -> tuple[str | None, str | None, str]:
    """Из messages [{user prompt}, {assistant}] достаёт (title, image_desc, meaning).

    title / image_desc — из user-промпта (что изображено, визуальный сигнал),
    meaning — из ответа assistant (смысл шутки, семантический сигнал).
    Если что-то не распарсилось, соответствующее поле = None (title/desc) или '' (meaning).
    """
    if not isinstance(messages, list):
        return None, None, ''
    user = messages[0].get('content', '') if len(messages) > 0 else ''
    asst = messages[1].get('content', '') if len(messages) > 1 else ''
    tm = TITLE_RE.search(user)
    dm = DESC_RE.search(user)
    title = norm(tm.group(1)) if tm else None
    image_desc = norm(dm.group(1)) if dm else None
    return title, image_desc, norm(asst)
