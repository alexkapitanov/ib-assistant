from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from models import AskRequest, AskResponse
from pipeline import search_vectors
from openai_helpers import clarify_question, get_embedding, rerank_chunks

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
    result = clarify_question(req.question)
    if "follow_up" in result:
        return AskResponse(answer=result["follow_up"])
    embedding = get_embedding(result["search_query"])
    hits = search_vectors(embedding, top_k=80)  # список строк-чанков
    reranked = rerank_chunks(hits, result["search_query"])
    top = reranked[:5]
    answer = (
        "\n".join(f"{r['score']}: {r['chunk'][:60]}…" for r in top)
        if top else "Фрагментов не найдено"
    )
    return AskResponse(answer=answer)
