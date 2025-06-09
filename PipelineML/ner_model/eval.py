import spacy
from seqeval.metrics import precision_score, recall_score, f1_score


def evaluate(nlp, texts, true_entities):
    preds, truths = [], []
    for text, gold in zip(texts, true_entities):
        doc = nlp(text)
        pred_labels = ['O'] * len(text.split())
        true_labels = ['O'] * len(text.split())

        for ent in doc.ents:
            for i in range(ent.start, ent.end):
                pred_labels[i] = "B-PRODUCT" if i == ent.start else "I-PRODUCT"

        for start, end, label in gold:
            for i in range(start, end):
                true_labels[i] = "B-PRODUCT" if i == start else "I-PRODUCT"

        preds.append(pred_labels)
        truths.append(true_labels)

    print("Precision:", precision_score(truths, preds))
    print("Recall:", recall_score(truths, preds))
    print("F1:", f1_score(truths, preds))
