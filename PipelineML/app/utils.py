import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import hashlib

def fetch_clean_text_with_links(url: str, max_pages: int = 20) -> str:

    visited_urls = set()
    collected_texts = []
    page_count = 0

    def is_internal(link, base_netloc):
        parsed = urlparse(link)
        return parsed.netloc == "" or parsed.netloc == base_netloc

    def clean_redundant_blocks(text: str) -> str:
        seen = set()
        filtered = []

        for paragraph in text.split("\n"):
            line = paragraph.strip()
            if not line or len(line) < 20:
                continue
            h = hashlib.md5(line.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                filtered.append(line)

        return "\n".join(filtered)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return ""

        base_url = url
        base_netloc = urlparse(url).netloc
        clean_url = urldefrag(url).url
        visited_urls.add(clean_url)

        # Main page
        main_text = trafilatura.extract(trafilatura.fetch_url(clean_url))
        if main_text:
            collected_texts.append(main_text)
            page_count += 1

        # Parse internal links
        soup = BeautifulSoup(response.text, "html.parser")
        links = [a.get("href") for a in soup.find_all("a", href=True)]
        internal_links = []

        for link in links:
            full_url = urldefrag(urljoin(base_url, link)).url
            if is_internal(full_url, base_netloc) and full_url not in visited_urls:
                internal_links.append(full_url)
                visited_urls.add(full_url)

        for link in internal_links:
            if page_count >= max_pages:
                break
            try:
                downloaded = trafilatura.fetch_url(link)
                if downloaded:
                    page_text = trafilatura.extract(downloaded)
                    if page_text:
                        collected_texts.append(page_text)
                        page_count += 1
            except Exception as e:
                print(f"[!] Skipped {link} — {e}")

    except Exception as e:
        print(f"[!] Base URL fetch failed: {e}")
        return ""

    full_text = "\n\n".join(collected_texts)
    return clean_redundant_blocks(full_text)
