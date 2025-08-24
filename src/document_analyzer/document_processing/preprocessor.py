import re


class TextPreprocessor:
    def __init__(self, chunk_size: int = 4000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean_text(self, text: str) -> str:
        """
        Cleans the text by removing extra whitespace and fixing encoding issues.
        """
        text = text.encode("utf-8", "ignore").decode("utf-8")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def chunk_text(self, text: str) -> list[str]:
        """
        Splits the text into chunks of a specified size with overlap.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks

    def preprocess(self, text: str) -> list[str]:
        """
        Runs the full preprocessing pipeline.
        """
        cleaned_text = self.clean_text(text)
        chunks = self.chunk_text(cleaned_text)
        return chunks
