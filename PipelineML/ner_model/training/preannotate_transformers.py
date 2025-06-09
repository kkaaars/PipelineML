import re
import os
import csv
from pathlib import Path
from typing import List, Tuple

PRODUCT_KEYWORDS = [
    "Mattress", "Bed", "Sofa", "Couch", "Chair", "Chaise", "Recliner", "Armchair",
    "Dining", "Table", "Set", "Stool", "Buffet", "Sideboard", "Cabinet", "Unit",
    "Stand", "Entertainment", "Coffee", "Desk", "Divider", "Shelf", "Storage", "Lounge"
]

EXCLUDE_PATTERNS = [
    r"Rated \d\.\d out of \d stars", r"Sale price", r"Regular price", r"RRP",
    r"Save \$?\d+", r"Now \$?\d+", r"\(\w+ \$\d+\)", r"\$\d+", r"Unit price", r"price"
]

def is_probable_product_line(line: str) -> bool:

    if len(line.strip()) < 20 or len(line.split()) > 25:
        return False
    if any(re.search(pat, line, re.IGNORECASE) for pat in EXCLUDE_PATTERNS):
        return False
    if any(kw.lower() in line.lower() for kw in PRODUCT_KEYWORDS):
        return True
    return False

def to_bio(text: str, label: str = "PRODUCT") -> Tuple[List[str], List[str]]:

    tokens = text.strip().split()
    if not tokens:
        return [], []
    labels = ["B-" + label] + ["I-" + label] * (len(tokens) - 1)
    return tokens, labels

def main():
    input_dir = "ner_model/data/raw/"
    output_csv = "ner_model/data/transformers_train.csv"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rows_written = 0
    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tokens", "labels"])

        for file_path in Path(input_dir).glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if is_probable_product_line(line):
                        tokens, labels = to_bio(line)
                        if tokens and labels:
                            writer.writerow([" ".join(tokens), " ".join(labels)])
                            rows_written += 1

    print(f"Done. Extracted {rows_written} product lines to {output_csv}")

if __name__ == "__main__":
    main()
