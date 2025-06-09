from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import torch

model_dir = "ner_model/model/checkpoint-375"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertForTokenClassification.from_pretrained(model_dir).to(device)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

text = "Pluto Bed & Mattress Package - Grey Queen"

results = ner_pipeline(text)
print("Prediction results:")
for r in results:
    print(r)
