import re

def preprocess_query(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)             # Remove HTML/XML tags
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)     # Remove control chars
    text = re.sub(r"\s+", " ", text)                # Normalize whitespace
    return text.strip()