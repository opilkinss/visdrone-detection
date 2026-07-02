"""
FastAPI backend для сервиса детекции объектов на снимках с дрона (VisDrone).
Оптимизирован для работы в 512MB RAM: модели загружаются лениво (lazy loading)
при первом запросе, а не при старте сервера.
"""

import io
import json
import os
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

# ──────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
CONFIG_PATH = os.path.join(BASE_DIR, "models_config.json")

MAX_FILE_SIZE_MB      = 15
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
DEFAULT_CONF          = 0.25
DEFAULT_IOU           = 0.7

# ──────────────────────────────────────────────────────────────────────────
# Приложение
# ──────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VisDrone Detection API",
    description="Детекция объектов на аэрофотоснимках с дрона",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────
# Конфиг
# ──────────────────────────────────────────────────────────────────────────

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"models_config.json не найден: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

CLASS_NAMES  = CONFIG["classes"]
MODELS_META  = {m["id"]: m for m in CONFIG["models"]}

# ──────────────────────────────────────────────────────────────────────────
# Lazy loading моделей — загружаем только при первом запросе
# ──────────────────────────────────────────────────────────────────────────

_loaded_models: dict = {}


def get_model(model_id: str):
    """
    Возвращает загруженную модель. Если ещё не загружена — загружает.
    Держит в памяти только одну модель одновременно чтобы уложиться в 512MB.
    """
    global _loaded_models

    if model_id in _loaded_models:
        return _loaded_models[model_id]

    if model_id not in MODELS_META:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестная модель '{model_id}'. Доступные: {list(MODELS_META.keys())}"
        )

    weights_path = os.path.join(MODELS_DIR, MODELS_META[model_id]["weights_file"])
    if not os.path.exists(weights_path):
        raise HTTPException(
            status_code=500,
            detail=f"Файл весов не найден: {weights_path}"
        )

    # Выгружаем предыдущую модель чтобы освободить RAM
    if _loaded_models:
        print(f"Выгружаем модель {list(_loaded_models.keys())} из RAM...")
        _loaded_models.clear()

        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Загружаем модель '{model_id}' из {weights_path}...")
    from ultralytics import YOLO
    model = YOLO(weights_path)
    _loaded_models[model_id] = model
    print(f"Модель '{model_id}' загружена.")

    return model


# ──────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────

def validate_upload(file: UploadFile, contents: bytes) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат: {file.content_type}. Допустимые: JPEG, PNG, WEBP."
        )
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Файл слишком большой: {size_mb:.1f} MB. Максимум: {MAX_FILE_SIZE_MB} MB."
        )


def run_inference(model_id: str, image: Image.Image, conf: float, iou: float):
    model = get_model(model_id)
    imgsz = MODELS_META[model_id]["imgsz"]

    start   = time.time()
    results = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    elapsed = (time.time() - start) * 1000

    detections = []
    r = results[0]
    if r.boxes is not None and len(r.boxes) > 0:
        for box, score, cls_id in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
        ):
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            detections.append({
                "class":      class_name,
                "class_id":   int(cls_id),
                "confidence": round(float(score), 4),
                "bbox":       [round(float(c), 2) for c in box],
            })

    return detections, elapsed


# ──────────────────────────────────────────────────────────────────────────
# Эндпоинты
# ──────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":        "ok",
        "service":       "VisDrone Detection API",
        "models_available": list(MODELS_META.keys()),
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/models")
def list_models():
    return {"models": list(MODELS_META.values())}


@app.get("/classes")
def list_classes():
    return {"classes": CLASS_NAMES}


@app.post("/predict")
async def predict(
    file:     UploadFile = File(...),
    model_id: str        = "accurate",
    conf:     float      = DEFAULT_CONF,
    iou:      float      = DEFAULT_IOU,
):
    if model_id not in MODELS_META:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестная модель '{model_id}'. Доступные: {list(MODELS_META.keys())}"
        )

    contents = await file.read()
    validate_upload(file, contents)

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Не удалось прочитать изображение. Файл повреждён или неподдерживаемый формат."
        )

    detections, elapsed_ms = run_inference(model_id, image, conf, iou)

    return JSONResponse({
        "model_used":        model_id,
        "model_name":        MODELS_META[model_id]["name"],
        "image_size":        list(image.size),
        "inference_ms":      round(elapsed_ms, 1),
        "detections_count":  len(detections),
        "detections":        detections,
    })


# ──────────────────────────────────────────────────────────────────────────
# Запуск напрямую: python main.py
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)