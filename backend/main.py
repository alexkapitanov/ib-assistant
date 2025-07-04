from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from backend.models import AskRequest, AskResponse
from backend.pipeline import search_vectors
from backend.openai_helpers import clarify_question, get_embedding

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
    hits = search_vectors(embedding)
    return AskResponse(answer=f"Найдено {len(hits)} релевантных фрагментов")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
