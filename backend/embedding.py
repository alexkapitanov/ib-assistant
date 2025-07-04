import os, openai
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "text-embedding-3-small"   # 1536-мерный
EMBED_DIM = 1536

def embed_text(text: str) -> list[float]:
    "Возвращает эмбеддинг длиной 1536"
    rsp = openai.Embedding.create(model=MODEL, input=text)
    return rsp["data"][0]["embedding"]
