"""
HTTP API za Flutter aplikaciju.

Pokretanje (iz korena projekta):
    python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

Tok:
    POST /api/predict          artist_name + audio fajl -> XGBoost predikcija (nekoliko sekundi)
    POST /api/analysis/{job}   Gemini + GPT + DeepSeek analiza istog fajla (do par minuta)
"""
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

try:
    # Koristi Windows/OS skladište sertifikata (npr. kada antivirus presreće HTTPS),
    # kako bi Gemini/GPT/DeepSeek/MusicBrainz pozivi prošli SSL verifikaciju.
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api import pipeline
from audio.utils.audio_utils import get_audiobox_data

ALLOWED_EXTENSIONS = {".mp3", ".wav"}  # GPT audio ulaz podržava samo mp3 i wav
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
JOB_TTL_SECONDS = 60 * 60

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _warm_up():
    """Učitava Audiobox model i listu izvođača unapred da prvi zahtev ne čeka."""
    pipeline.model_info()
    pipeline._artists_df()
    warmup_file = Path(tempfile.gettempdir()) / "hit_song_warmup.wav"
    import numpy as np
    import soundfile as sf
    sf.write(warmup_file, np.zeros(22050, dtype=np.float32), 22050)
    get_audiobox_data(str(warmup_file))
    print("[api] Models warmed up and ready.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="Hit Song Predictor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cleanup_old_jobs():
    now = time.time()
    with JOBS_LOCK:
        expired = [jid for jid, job in JOBS.items() if now - job["created"] > JOB_TTL_SECONDS]
        for jid in expired:
            shutil.rmtree(JOBS.pop(jid)["dir"], ignore_errors=True)


@app.get("/api/health")
def health():
    return {"status": "ok", **pipeline.model_info()}


@app.get("/api/artists")
def artists(q: str = ""):
    return pipeline.search_artists(q)


@app.post("/api/predict")
def predict(artist_name: str = Form(...), audio: UploadFile = File(...)):
    _cleanup_old_jobs()

    artist_name = artist_name.strip()
    if not artist_name:
        raise HTTPException(400, "Artist name is required.")
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only .mp3 and .wav files are supported.")

    job_id = uuid.uuid4().hex
    job_dir = Path(tempfile.mkdtemp(prefix=f"hit_song_{job_id[:8]}_"))
    audio_path = job_dir / f"upload{suffix}"
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    if audio_path.stat().st_size > MAX_UPLOAD_BYTES:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(413, "File is larger than 50 MB.")

    started = time.time()
    try:
        result = pipeline.run_prediction(str(audio_path), str(job_dir / "window.wav"), artist_name)
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"Prediction failed: {e}")

    with JOBS_LOCK:
        JOBS[job_id] = {
            "created": time.time(),
            "dir": job_dir,
            "audio_path": str(audio_path),
            "artist_name": artist_name,
            "librosa_features": result.pop("_librosa_features"),
            "talent_data": result.pop("_talent_data"),
        }

    return {
        "job_id": job_id,
        "file_name": audio.filename,
        "elapsed_sec": round(time.time() - started, 2),
        **result,
    }


@app.post("/api/analysis/{job_id}")
def analysis(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job. Please run the prediction again.")

    started = time.time()
    result = pipeline.run_llm_analysis(
        job["audio_path"], job["artist_name"], job["librosa_features"], job["talent_data"]
    )
    return {"job_id": job_id, "elapsed_sec": round(time.time() - started, 2), **result}
