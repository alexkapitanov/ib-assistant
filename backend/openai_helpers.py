import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "clarify.txt")


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
