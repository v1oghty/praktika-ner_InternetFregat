# -*- coding: utf-8 -*-
"""
Обучение нейросетевой NER-модели (RuBERT) на gold_dataset.json
с корректным выравниванием меток под BPE-токенизатор.
"""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# ============================================================
# 1. Загрузка датасета
# ============================================================
with open("gold_dataset_final.json", encoding="utf-8") as f:
    gold = json.load(f)

# Метки
LABEL_LIST = ["O", "B-CAD_NUMBER", "I-CAD_NUMBER", "B-ADDRESS", "I-ADDRESS", "B-PERSON", "I-PERSON"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

# ============================================================
# 2. Подготовка данных
# ============================================================
def prepare_data(gold_items):
    """Преобразует JSON в тексты и посимвольные BIO-метки."""
    texts = []
    all_char_labels = []

    for item in gold_items:
        text = item["text"]
        entities = sorted(item["entities"], key=lambda e: e["start"])

        char_labels = ["O"] * len(text)

        for ent in entities:
            label = ent["label"]
            start = ent["start"]
            end = ent["end"]

            if end > len(text):
                end = len(text)

            if start >= len(text):
                continue

            char_labels[start] = f"B-{label}"
            for i in range(start + 1, end):
                char_labels[i] = f"I-{label}"

        texts.append(text)
        all_char_labels.append(char_labels)

    return texts, all_char_labels


def tokenize_and_align(texts, char_labels_list, tokenizer, max_length=256):
    """Токенизирует и выравнивает метки под BPE-токенизатор."""
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
    )

    all_labels = []

    for i, char_labels in enumerate(char_labels_list):
        offsets = tokenized["offset_mapping"][i]
        labels = []

        for offset in offsets:
            start = int(offset[0])
            end = int(offset[1])

            # Специальные токены [CLS]=[0,0], [SEP]=[0,0], [PAD]=[0,0]
            if start == 0 and end == 0:
                labels.append(-100)
            elif start < len(char_labels):
                token_label = char_labels[start]
                labels.append(LABEL2ID.get(token_label, 0))
            else:
                labels.append(-100)

        all_labels.append(labels)

    tokenized["labels"] = all_labels
    return tokenized


# ============================================================
# 3. Готовим датасет
# ============================================================
texts, char_labels_list = prepare_data(gold)

MODEL_NAME = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenized_data = tokenize_and_align(texts, char_labels_list, tokenizer)

dataset = Dataset.from_dict({
    "input_ids": tokenized_data["input_ids"],
    "attention_mask": tokenized_data["attention_mask"],
    "labels": tokenized_data["labels"],
})

dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(f"Train: {len(dataset['train'])} примеров")
print(f"Test:  {len(dataset['test'])} примеров")

# ============================================================
# 4. Загружаем модель
# ============================================================
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# ============================================================
# 5. Обучение
# ============================================================
training_args = TrainingArguments(
    output_dir="./legal_ner_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=30,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

print("Начинаю обучение...")
trainer.train()

model.save_pretrained("./legal_ner_model")
tokenizer.save_pretrained("./legal_ner_model")
print("Модель сохранена в ./legal_ner_model")