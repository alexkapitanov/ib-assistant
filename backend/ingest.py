import os, io, subprocess, tempfile, uuid, json
from typing import List
import pdfplumber
from PIL import Image
import easyocr
from whisper import load_model as load_whisper
from backend.openai_helpers import get_embedding
from backend.pipeline import qdrant_client

reader = easyocr.Reader(["en", "ru"], gpu=False)
whisper_model = load_whisper("base")

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
    """Создаёт эмбеддинги и кладёт их в Qdrant."""
    match filetype:
        case "pdf": chunks = process_pdf(path)
        case "png" | "jpg" | "jpeg": chunks = process_image(path)
        case "mp4" | "mov": chunks = process_video(path)
        case _: return
    vectors = []
    for idx, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        vectors.append(
            {
                "id": f"{file_id}_{idx}",
                "vector": emb,
                "payload": {"text": chunk, "file_id": file_id, "index": idx},
            }
        )
    if vectors:
        qdrant_client.upsert(
            collection_name="ib-assistant",
            points=vectors,
        )
