from transformers import pipeline

extractor = pipeline(
    "ner",
    model="ner_model/model/checkpoint-375",
    tokenizer="ner_model/model/checkpoint-375",
    aggregation_strategy="simple"
)

def extract_products(text: str):
    entities = extractor(text)
    products = [ent["word"] for ent in entities if ent["entity_group"] == "PRODUCT"]
    return products
