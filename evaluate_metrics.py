# -*- coding: utf-8 -*-
"""
Расчёт метрик Precision / Recall / F1-score для сравнения подходов
извлечения именованных сущностей (regex-модуль vs нейросетевая модель)
на эталонном (gold) датасете.

Особенности реализации:
  - Exact match: полное совпадение span'а и метки
  - Partial match: пересечение span'ов при совпадении метки
  - Поддержка двух предиктов: rule-based и нейросетевого
  - Сравнительная таблица в конце
"""

import json
import torch
from collections import defaultdict
from typing import List, Tuple, Callable

from extraction_module import extract_all, Entity
from transformers import AutoTokenizer, AutoModelForTokenClassification


# ============================================================================
# Глобальная загрузка нейросетевой модели (один раз)
# ============================================================================
_MODEL = None
_TOKENIZER = None
_LABEL_LIST = ["O", "B-CAD_NUMBER", "I-CAD_NUMBER", "B-ADDRESS", "I-ADDRESS", "B-PERSON", "I-PERSON"]
_ID2LABEL = {i: l for i, l in enumerate(_LABEL_LIST)}


def _load_model():
    """Загружает модель один раз при первом вызове."""
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        print("Загрузка нейросетевой модели...")
        _TOKENIZER = AutoTokenizer.from_pretrained("./legal_ner_model")
        _MODEL = AutoModelForTokenClassification.from_pretrained("./legal_ner_model")
        _MODEL.eval()
        print("Модель загружена.\n")


def neural_predict(text: str) -> List[Entity]:
    """Реальный нейросетевой предикт через обученную модель."""
    _load_model()

    # Токенизация с offset_mapping
    inputs = _TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")[0]

    # Прогон через модель
    with torch.no_grad():
        outputs = _MODEL(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)[0]

    # Собираем сущности
    entities = []
    current_entity = None  # [start_char, end_char, label]

    for i, pred_id in enumerate(predictions):
        pred_id = pred_id.item()
        if pred_id == -100:
            continue

        label = _ID2LABEL.get(pred_id, "O")
        start_char = int(offset_mapping[i][0])
        end_char = int(offset_mapping[i][1])

        # Пропускаем токены с нулевым span
        if start_char == end_char:
            continue

        if label.startswith("B-"):
            # Сохраняем предыдущую сущность
            if current_entity is not None:
                entities.append(Entity(
                    current_entity[0], current_entity[1],
                    text[current_entity[0]:current_entity[1]], current_entity[2]
                ))
            # Начинаем новую
            current_entity = [start_char, end_char, label[2:]]

        elif label.startswith("I-") and current_entity is not None:
            if label[2:] == current_entity[2]:
                current_entity[1] = end_char
            else:
                entities.append(Entity(
                    current_entity[0], current_entity[1],
                    text[current_entity[0]:current_entity[1]], current_entity[2]
                ))
                current_entity = None

        else:
            if current_entity is not None:
                entities.append(Entity(
                    current_entity[0], current_entity[1],
                    text[current_entity[0]:current_entity[1]], current_entity[2]
                ))
                current_entity = None

    # Не забываем последнюю сущность
    if current_entity is not None:
        entities.append(Entity(
            current_entity[0], current_entity[1],
            text[current_entity[0]:current_entity[1]], current_entity[2]
        ))

    # Удаляем дубликаты
    seen = set()
    unique = []
    for e in entities:
        key = (e.start, e.end, e.label)
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


# ============================================================================
# Загрузка эталонного датасета
# ============================================================================
def load_gold(path: str) -> list:
    """Загрузка эталонного датасета."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Вспомогательные функции
# ============================================================================
def spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Проверка пересечения двух интервалов."""
    return a[0] < b[1] and b[0] < a[1]


def calculate_metrics(
    gold_items: list,
    predict_fn: Callable[[str], List[Entity]],
    mode: str = "exact"
) -> dict:
    """
    Расчёт метрик для заданного предикта.
    """
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for item in gold_items:
        text = item["text"]
        gold_ents = [(e["start"], e["end"], e["label"]) for e in item["entities"]]
        pred_ents = [(e.start, e.end, e.label) for e in predict_fn(text)]

        matched_gold = set()
        matched_pred = set()

        for pi, (ps, pe, plabel) in enumerate(pred_ents):
            for gi, (gs, ge, glabel) in enumerate(gold_ents):
                if gi in matched_gold or plabel != glabel:
                    continue

                if mode == "exact":
                    is_match = (ps == gs and pe == ge)
                else:
                    is_match = spans_overlap((ps, pe), (gs, ge))

                if is_match:
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    stats[plabel]["tp"] += 1
                    break

        for pi, (ps, pe, plabel) in enumerate(pred_ents):
            if pi not in matched_pred:
                stats[plabel]["fp"] += 1

        for gi, (gs, ge, glabel) in enumerate(gold_ents):
            if gi not in matched_gold:
                stats[glabel]["fn"] += 1

    result = {}
    total_tp = total_fp = total_fn = 0

    all_labels = sorted(set(
        list(stats.keys()) +
        [e["label"] for item in gold_items for e in item["entities"]]
    ))

    for label in all_labels:
        s = stats.get(label, {"tp": 0, "fp": 0, "fn": 0})
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        result[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    result["OVERALL"] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
    }

    return result


# ============================================================================
# Вывод результатов
# ============================================================================
def print_report(title: str, result: dict):
    """Вывод отчёта в табличном виде."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Тип':<14}{'Precision':<12}{'Recall':<12}{'F1-score':<12}{'TP':<6}{'FP':<6}{'FN':<6}")
    print(f"{'-'*70}")

    labels = [l for l in result.keys() if l != "OVERALL"] + (["OVERALL"] if "OVERALL" in result else [])
    for label in labels:
        if label not in result:
            continue
        m = result[label]
        print(f"{label:<14}{m['precision']:<12}{m['recall']:<12}{m['f1']:<12}{m['tp']:<6}{m['fp']:<6}{m['fn']:<6}")
    print(f"{'='*70}")


def print_comparison(regex_result: dict, neural_result: dict):
    """Сравнительная таблица двух подходов."""
    print(f"\n{'='*80}")
    print(f"  СРАВНИТЕЛЬНЫЙ АНАЛИЗ: Regex-модуль vs Нейросеть (RuBERT)")
    print(f"{'='*80}")
    print(f"{'Класс':<14}{'Regex F1':<14}{'Neural F1':<14}{'Разница':<14}{'Победитель':<14}")
    print(f"{'-'*70}")

    all_labels = sorted(set(list(regex_result.keys()) + list(neural_result.keys())))
    for label in all_labels:
        if label == "OVERALL":
            continue
        regex_f1 = regex_result.get(label, {}).get("f1", 0)
        neural_f1 = neural_result.get(label, {}).get("f1", 0)
        diff = round(neural_f1 - regex_f1, 4)
        winner = "Нейросеть" if diff > 0 else ("Regex" if diff < 0 else "Ничья")
        print(f"{label:<14}{regex_f1:<14}{neural_f1:<14}{diff:<14}{winner:<14}")

    regex_ov = regex_result.get("OVERALL", {}).get("f1", 0)
    neural_ov = neural_result.get("OVERALL", {}).get("f1", 0)
    diff_ov = round(neural_ov - regex_ov, 4)
    winner_ov = "Нейросеть" if diff_ov > 0 else ("Regex" if diff_ov < 0 else "Ничья")
    print(f"{'-'*70}")
    print(f"{'OVERALL':<14}{regex_ov:<14}{neural_ov:<14}{diff_ov:<14}{winner_ov:<14}")
    print(f"{'='*80}")


# ============================================================================
# Главный блок
# ============================================================================
if __name__ == "__main__":
    gold = load_gold("gold_dataset_final.json")
    print(f"Загружено примеров: {len(gold)}")
    print(f"Сущностей в датасете: {sum(len(item['entities']) for item in gold)}")

    # Regex
    regex_exact = calculate_metrics(gold, extract_all, mode="exact")
    regex_partial = calculate_metrics(gold, extract_all, mode="partial")

    print_report("Regex-модуль — точное совпадение (exact match)", regex_exact)
    print_report("Regex-модуль — пересечение спанов (partial match)", regex_partial)

    # Нейросеть
    neural_exact = calculate_metrics(gold, neural_predict, mode="exact")

    print_report("Нейросеть (RuBERT) — точное совпадение", neural_exact)

    # Сравнение
    print_comparison(regex_exact, neural_exact)

    # Вывод
    print(f"""
{'='*80}
  ВЫВОД
{'='*80}
Проведено сравнение двух подходов к извлечению именованных сущностей
из текстов нормативно-правовых актов:

1. Regex-модуль:
   - Показывает высокую точность на стандартизированных данных
   - Не обрабатывает опечатки и нетипичные форматы
   - Не требует обучения, работает мгновенно
   - F1 (exact): {regex_exact['OVERALL']['f1']}
   - F1 (partial): {regex_partial['OVERALL']['f1']}

2. Нейросетевой подход (RuBERT, дообученный на 18 примерах):
   - Компенсирует недостатки regex за счёт обучения на контексте
   - Требует размеченного датасета и времени на обучение
   - F1 (exact): {neural_exact['OVERALL']['f1']}

Вывод: {'Нейросетевой подход показал более высокие метрики' if neural_exact['OVERALL']['f1'] > regex_exact['OVERALL']['f1'] else 'Regex-подход показал более высокие метрики на данном датасете'}.
{'='*80}
""")