from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from models import AskRequest, AskResponse
from pipeline import search_vectors
from openai_helpers import clarify_question, get_embedding, rerank_chunks
from uuid import uuid4
from ingest import ingest_file
import boto3, os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# клиент MinIO
minio = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
    aws_access_key_id=os.getenv("MINIO_ROOT_USER", "minio"),
    aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "minio123"),
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

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Принимает PDF/PNG/MP4, сохраняет в MinIO и индексирует."""
    file_id = str(uuid4())
    ext = file.filename.split(".")[-1].lower()
    tmp_path = f"/tmp/{file_id}.{ext}"

    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # кладём в MinIO (bucket ib-files должен существовать)
    minio.upload_file(tmp_path, "ib-files", f"{file_id}.{ext}")

    # индексируем
    ingest_file(tmp_path, ext, file_id)

    return {"status": "uploaded", "file_id": file_id, "chunks_indexed": "done"}
