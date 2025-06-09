import os
import pandas as pd
from app.utils import fetch_clean_text_with_links
import requests

RAW_DIR = "ner_model/data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

url_df = pd.read_csv("ner_model/data/URL_list.csv")
urls = url_df.iloc[:, 0].dropna().unique().tolist()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

success_count = 0
for i, url in enumerate(urls[:100]):
    print(f"[{i+1}/{len(urls[:100])}] Checking URL: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)

        if response.status_code != 200:
            print(f"[!] Skipped (status code {response.status_code}): {url}")
            continue

        if not response.text or len(response.text.strip()) < 100:
            print(f"[!] Skipped (empty content): {url}")
            continue

        text = fetch_clean_text_with_links(url)
        if not text or len(text.strip()) < 100:
            print(f"[!] Skipped (cleaned content empty): {url}")
            continue

        filename = f"{RAW_DIR}/page_{i:03}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)

        success_count += 1
        print(f"[✓] Saved: {filename}")

    except Exception as e:
        print(f"[X] Failed to process {url} — {e}")

print(f"\n✔ Done. Successfully saved {success_count} files.")
