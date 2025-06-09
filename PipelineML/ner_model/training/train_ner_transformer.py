import os
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split
import torch

ANNOTATED_DATA_PATH = "ner_model/data/transformers_train.csv"
MODEL_DIR = "ner_model/pretrained/bert-base-uncased"
OUTPUT_DIR = "ner_model/model"

label_list = ["O", "B-PRODUCT", "I-PRODUCT"]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}


def parse_row(row):
    tokens = str(row["tokens"]).split()
    labels = str(row["labels"]).split()

    if len(tokens) != len(labels):
        return None

    return {"tokens": tokens, "labels": labels}


df = pd.read_csv(ANNOTATED_DATA_PATH, dtype={"tokens": str, "labels": str})

parsed = df.apply(parse_row, axis=1).dropna().tolist()

dataset = Dataset.from_list(parsed)

split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)


def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    word_ids = tokenized.word_ids()
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(label_to_id[example["labels"][word_idx]])
        else:
            aligned_labels.append(label_to_id[example["labels"][word_idx]])
        previous_word_idx = word_idx
    tokenized["labels"] = aligned_labels
    return tokenized


train_dataset = train_dataset.map(tokenize_and_align_labels, batched=False)
eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=False)

model = BertForTokenClassification.from_pretrained(MODEL_DIR, num_labels=len(label_list))
model.config.id2label = {0: "O", 1: "B-PRODUCT", 2: "I-PRODUCT"}
model.config.label2id = {"O": 0, "B-PRODUCT": 1, "I-PRODUCT": 2}


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="logs",
    logging_steps=50,
    do_eval=True,
    do_train=True
)


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)


trainer.train()


trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
