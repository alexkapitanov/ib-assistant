import os, io, subprocess, tempfile, uuid, json
from typing import List
import pdfplumber
from PIL import Image
import easyocr
from whisper import load_model as load_whisper
from backend.embedding import embed_text, EMBED_DIM
from backend.pipeline import qdrant
import boto3, os, botocore
from qdrant_client import QdrantClient, models
from backend.qdrant_utils import get_client, COLLECTION

reader = easyocr.Reader(["en", "ru"], gpu=False)
whisper_model = load_whisper("base")

def ensure_minio_bucket(bucket_name: str = "ib-files"):
    """Создаёт bucket в MinIO, если его ещё нет (идемпотентно)."""
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER", "minio"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "minio123"),
    )
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"[MinIO] bucket '{bucket_name}' ok")
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            s3.create_bucket(Bucket=bucket_name)
            print(f"[MinIO] bucket '{bucket_name}' created ✔")
        else:
            raise

def chunk_text(text: str, max_len: int = 256) -> List[str]:
    """Режем текст на куски ≈256 токенов (простое деление по предложениям)."""
    sentences, chunk, out = text.split("."), [], []
    for s in sentences:
        if len(" ".join(chunk + [s]).split()) > max_len:
            out.append(".".join(chunk).strip())
            chunk = []
        chunk.append(s)
    if chunk:
        out.append(".".join(chunk).strip())
    return out

def process_pdf(path: str) -> List[str]:
    txt = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt += page.extract_text() + "\n"
    return chunk_text(txt)

def process_image(path: str) -> List[str]:
    result = reader.readtext(path, detail=0)
    return chunk_text(" ".join(result))

def extract_frames(video_path: str, out_dir: str):
    """Сохраняет 1 кадр в сек.; adjust as needed."""
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", "fps=1", f"{out_dir}/frame_%04d.png"],
        check=True,
        capture_output=True,
    )

def process_video(path: str) -> List[str]:
    with tempfile.TemporaryDirectory() as tmp:
        extract_frames(path, tmp)
        chunks = []
        for frame in sorted(os.listdir(tmp)):
            frame_path = os.path.join(tmp, frame)
            chunks.extend(process_image(frame_path))
        # + Whisper-audio
        audio = whisper_model.transcribe(path)["text"]
        chunks.extend(chunk_text(audio))
        return chunks

def ingest_file(path: str, filetype: str, file_id: str):
    """Создаёт эмбеддинги и кладёт их в Qdrant (docs)."""
    match filetype:
        case "pdf": chunks = process_pdf(path)
        case "png" | "jpg" | "jpeg": chunks = process_image(path)
        case "mp4" | "mov": chunks = process_video(path)
        case _: return
    client = get_client()
    points = []
    for idx, chunk in enumerate(chunks):
        vector = embed_text(chunk)
        point_id = str(uuid.uuid4())
        payload = {"text": chunk, "file_id": file_id, "index": idx}
        points.append(
            models.PointStruct(id=point_id, vector=vector, payload=payload)
        )
    if points:
        client.upsert(collection_name=COLLECTION, wait=True, points=points)
