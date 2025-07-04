from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from backend.models import AskRequest, AskResponse
import openai
from backend.pipeline import search_vectors

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
async def ask(request: Request):
    data = await request.json()
    req = AskRequest(**data)
    # Получаем embedding через OpenAI
    embedding_response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=req.question
    )
    embedding = embedding_response["data"][0]["embedding"]
    # Поиск фрагментов
    results = search_vectors(embedding, top_k=20)
    answer = f"Найдено {len(results)} фрагментов"
    return AskResponse(answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
