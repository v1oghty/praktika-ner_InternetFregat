# -*- coding: utf-8 -*-
"""
Модуль извлечения именованных сущностей из текстов НПА
на основе регулярных выражений и синтаксических шаблонов.

Поддерживаемые типы сущностей:
  - CAD_NUMBER  — кадастровые номера
  - ADDRESS     — адреса
  - PERSON      — ФИО

Отличие от альтернативного подхода: данный модуль НЕ использует
морфологическую нормализацию и НЕ обрабатывает опечатки в кадастровых
номерах (это осознанное ограничение — для сравнения с нейросетевым
подходом, который должен компенсировать эти недостатки).

Каждая функция возвращает список Entity.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Entity:
    start: int
    end: int
    text: str
    label: str

    def as_tuple(self):
        return (self.start, self.end, self.text, self.label)


# ---------------------------------------------------------------------------
# 1. КАДАСТРОВЫЕ НОМЕРА
# ---------------------------------------------------------------------------
# Канонический формат: AA:BB:CCCCCCC:DD
# Поддерживаемые разделители: двоеточие, тире
# Опечатки и слитные номера НЕ обрабатываются — это зона нейросети

_CADASTRAL_PATTERN = re.compile(
    r"""
    (?<!\d)
    \d{2,3}               # регион (2-3 цифры)
    [:\-]                  # разделитель
    \d{1,2}               # район (1-2 цифры)
    [:\-]                  # разделитель
    \d{5,7}               # квартал (5-7 цифр)
    [:\-]                  # разделитель
    \d{1,10}              # номер объекта (1-10 цифр)
    (?!\d)
    """,
    re.VERBOSE,
)


def extract_cadastral_numbers(text: str) -> List[Entity]:
    entities = []
    for m in _CADASTRAL_PATTERN.finditer(text):
        raw = m.group(0)
        entities.append(Entity(m.start(), m.end(), raw, "CAD_NUMBER"))
    return entities


# ---------------------------------------------------------------------------
# 2. АДРЕСА
# ---------------------------------------------------------------------------
# Собираем адрес как последовательность компонентов.
# Каждый компонент ищется отдельным паттерном, затем близко расположенные
# компоненты объединяются в цепочку (с окном в 3 слова между компонентами).

# Словари маркеров (шире, чем в изначальной версии, но без экзотики вроде снт/влд)
_REGION_MARKERS = r"(?:г\.о\.|городской округ|г\.|город|пос\.|посёлок|поселок|с\.|село|д\.|деревня|пгт\.?)"
_STREET_MARKERS = r"(?:ул\.|улица|пр-кт|проспект|пр\.|пер\.|переулок|ш\.|шоссе|наб\.|набережная|пл\.|площадь)"
_HOUSE_MARKERS = r"(?:д\.|дом|стр\.|строение|корп\.|корпус)"
_FLAT_MARKERS = r"(?:кв\.|квартира|оф\.|офис|пом\.|помещение)"

# Отдельные паттерны для каждого компонента
_COMPONENT_PATTERNS = [
    # Населённый пункт: "г. Москва", "г.о. Химки", "пос. Дубки"
    re.compile(rf"{_REGION_MARKERS}\s+[А-ЯЁ][а-яё\-]+(\s+[А-ЯЁ][а-яё\-]+)?", re.IGNORECASE),
    # Улица: "ул. Ленина", "пр-кт Мира", "наб. Обводного канала"
    re.compile(rf"{_STREET_MARKERS}\s+[А-ЯЁа-яё0-9][а-яё0-9\-\.\s]{{1,30}}?(?=,|\s+(?:{_HOUSE_MARKERS})|\s*$)", re.IGNORECASE),
    # Дом: "д. 15", "дом 15к2", "стр. 15/14"
    re.compile(rf"{_HOUSE_MARKERS}\s+\d+[А-Яа-я]?(?:/\d+)?", re.IGNORECASE),
    # Квартира/офис: "кв. 5", "офис 12", "пом. IV"
    re.compile(rf"{_FLAT_MARKERS}\s+\d+[А-Яа-я]?", re.IGNORECASE),
]


def extract_addresses(text: str) -> List[Entity]:
    # Находим все компоненты всех типов
    all_matches = []
    for pat in _COMPONENT_PATTERNS:
        for m in pat.finditer(text):
            all_matches.append((m.start(), m.end(), m.group(0)))

    if not all_matches:
        return []

    # Сортируем по позиции в тексте
    all_matches.sort(key=lambda x: x[0])

    # Группируем близко расположенные компоненты в адреса
    # Максимальный разрыв между компонентами — 25 символов (~3-4 слова)
    MAX_GAP = 25

    entities = []
    current_chain = [all_matches[0]]

    for i in range(1, len(all_matches)):
        prev_end = current_chain[-1][1]
        curr_start = all_matches[i][0]

        # Проверяем, что между компонентами нет знаков, разрывающих адрес
        gap_text = text[prev_end:curr_start]

        if len(gap_text) <= MAX_GAP and not re.search(r'[\.!\?]{2,}', gap_text):
            # Продолжаем цепочку
            current_chain.append(all_matches[i])
        else:
            # Завершаем цепочку и начинаем новую
            if len(current_chain) >= 2:  # минимум 2 компонента для адреса
                start = current_chain[0][0]
                end = current_chain[-1][1]
                addr_text = text[start:end].strip().rstrip(',')
                entities.append(Entity(start, end, addr_text, "ADDRESS"))
            current_chain = [all_matches[i]]

    # Не забываем последнюю цепочку
    if len(current_chain) >= 2:
        start = current_chain[0][0]
        end = current_chain[-1][1]
        addr_text = text[start:end].strip().rstrip(',')
        entities.append(Entity(start, end, addr_text, "ADDRESS"))

    return entities


# ---------------------------------------------------------------------------
# 3. ФИО
# ---------------------------------------------------------------------------
# Варианты:
#   Иванов Иван Иванович           (полное, именительный)
#   Иванову Ивану Ивановичу        (дательный падеж)
#   Иванов И.И. / И.И. Иванов      (с инициалами)
#   Петрова Мария Александровна    (женский род)
#   Петровой Марии Александровне   (женский, дательный)

_CAPWORD = r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"  # слово с заглавной

# Полное ФИО (три слова с заглавной, последнее — отчество)
_FIO_FULL = re.compile(
    rf"""
    \b{_CAPWORD}\s+{_CAPWORD}\s+{_CAPWORD}\b
    (?=\s|,|\.|$)
    """,
    re.VERBOSE,
)

# ФИО с инициалами: Фамилия И.О. или И.О. Фамилия
_FIO_INITIALS = re.compile(
    rf"""
    \b{_CAPWORD}\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.     # Фамилия И.О.
    |
    \b[А-ЯЁ]\.\s?[А-ЯЁ]\.\s+{_CAPWORD}\b   # И.О. Фамилия
    """,
    re.VERBOSE,
)


def extract_persons(text: str) -> List[Entity]:
    entities = []
    seen_spans = set()

    # Сначала ищем полные ФИО
    for m in _FIO_FULL.finditer(text):
        span = (m.start(), m.end())
        if span not in seen_spans:
            entities.append(Entity(m.start(), m.end(), m.group(0), "PERSON"))
            seen_spans.add(span)

    # Потом инициалы (но не внутри уже найденных полных ФИО)
    for m in _FIO_INITIALS.finditer(text):
        span = (m.start(), m.end())
        # Проверяем, не пересекается ли с уже найденным
        overlaps_existing = any(
            span[0] < existing[1] and existing[0] < span[1]
            for existing in seen_spans
        )
        if not overlaps_existing:
            entities.append(Entity(m.start(), m.end(), m.group(0), "PERSON"))
            seen_spans.add(span)

    return entities


# ---------------------------------------------------------------------------
# Общая точка входа
# ---------------------------------------------------------------------------
def extract_all(text: str) -> List[Entity]:
    """
    Извлекает все сущности из текста.
    Сущности сортируются по позиции в тексте.
    При пересечении спанов приоритет:
      1. Более длинный span (полные ФИО > инициалы)
      2. Первый по порядку
    """
    all_ents = (
        extract_cadastral_numbers(text)
        + extract_addresses(text)
        + extract_persons(text)
    )

    # Сортировка и удаление пересекающихся спанов
    all_ents.sort(key=lambda e: (e.start, -e.end))  # сначала длинные span'ы

    filtered = []
    occupied = []  # список занятых интервалов

    for ent in all_ents:
        overlaps = any(
            ent.start < occ_end and occ_start < ent.end
            for occ_start, occ_end in occupied
        )
        if not overlaps:
            filtered.append(ent)
            occupied.append((ent.start, ent.end))

    filtered.sort(key=lambda e: e.start)
    return filtered


# ---------------------------------------------------------------------------
# Тестовый запуск
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        'Администрация г.о. Мытищи постановляет предоставить Иванову И.И. '
        'земельный участок с кадастровым номером 50:21:0010102:123, '
        'расположенный по адресу: Московская область, г. Мытищи, ул. Летная, д. 15/14, кв. 5. '
        'Право собственности зарегистрировано за Петровой Марией Александровной. '
        'Представитель — А.А. Смирнов. '
        'Также рассмотрен участок 77-08-0005001-456.'
    )

    print("Извлечённые сущности:\n")
    for e in extract_all(sample):
        print(f"  [{e.label}] {repr(e.text)} (позиции {e.start}-{e.end})")