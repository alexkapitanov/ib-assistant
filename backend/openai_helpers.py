import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "clarify.txt")
RERANK_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "rerank.txt")


def get_embedding(text: str) -> list:
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    return response["data"][0]["embedding"]


def clarify_question(question: str) -> dict:
    with open(PROMPT_PATH, encoding="utf-8") as f:
        prompt = f.read().strip()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=128
    )
    text = response["choices"][0]["message"]["content"].strip()
    if text.startswith("Follow-up вопрос:"):
        return {"follow_up": text[len("Follow-up вопрос:"):].strip()}
    elif text.startswith("Поисковый запрос:"):
        return {"search_query": text[len("Поисковый запрос:"):].strip()}
    else:
        return {"raw": text}


def rerank_chunks(chunks: list[str], search_query: str) -> list[dict]:
    """
    Возвращает отсортированный по убыванию score список:
    { "chunk": str, "score": int, "comment": str }
    Использует prompt из backend/prompts/rerank.txt.
    Модель: gpt-4o-mini.
    Если chunks пустой → вернуть [].
    """
    if not chunks:
        return []
    with open(RERANK_PROMPT_PATH, encoding="utf-8") as f:
        prompt = f.read().strip()
    # Формируем сообщение для LLM
    user_content = f"Поисковый запрос: {search_query}\n\nКандидаты:\n" + "\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(chunks))
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=1024
    )
    # Ожидаем, что LLM вернёт JSON-массив или список в формате:
    # [ { "chunk": ..., "score": ..., "comment": ... }, ... ]
    import json
    import re
    text = response["choices"][0]["message"]["content"].strip()
    # Пытаемся извлечь JSON из ответа
    match = re.search(r'(\[.*\])', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            # Сортируем по score по убыванию
            return sorted(result, key=lambda x: x.get("score", 0), reverse=True)
        except Exception:
            pass
    # Если не удалось — возвращаем пустой список
    return []
