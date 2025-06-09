from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import os

model_dir = os.path.abspath("ner_model/model/checkpoint-375")

print(f"Downloading model from: {model_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True).to(device)

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=-1)


def extract_products(text: str):
    results = nlp(text)
    print("Raw predictions:")
    for r in results:
        print(r)

    raw_products = [r["word"] for r in results if r.get("entity_group") == "PRODUCT"]

    print("All predicted products:")
    for p in raw_products:
        print(f" - {p}")

    return raw_products

