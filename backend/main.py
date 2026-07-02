"""
FastAPI backend для сервиса детекции объектов на снимках с дрона (VisDrone).
Обслуживает две модели: быструю (imgsz=640) и точную (imgsz=1280).
"""

import io
import json
import os
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_PATH = os.path.join(BASE_DIR, "models_config.json")

MAX_FILE_SIZE_MB = 15
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7

# ──────────────────────────────────────────────────────────────────────────
# Приложение
# ──────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VisDrone Detection API",
    description="Детекция объектов (пешеходы, машины, велосипеды и т.д.) на снимках с дрона",
    version="1.0.0",
)

# Разрешаем запросы с фронтенда (Gradio может быть на другом порту/домене)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────
# Загрузка конфигурации и моделей при старте
# ──────────────────────────────────────────────────────────────────────────

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"models_config.json не найден по пути {CONFIG_PATH}. "
        f"Убедитесь, что файл лежит рядом с main.py."
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

MODELS = {}

for model_meta in CONFIG["models"]:
    weights_path = os.path.join(MODELS_DIR, model_meta["weights_file"])
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Файл весов не найден: {weights_path}. "
            f"Проверьте, что веса скопированы в папку backend/models/."
        )

    print(f"Загрузка модели '{model_meta['id']}' из {weights_path} ...")
    loaded_model = YOLO(weights_path)

    MODELS[model_meta["id"]] = {
        "model": loaded_model,
        "imgsz": model_meta["imgsz"],
        "meta": model_meta,
    }
    print(f"  готово: {model_meta['name']}")

CLASS_NAMES = CONFIG["classes"]

print(f"\nВсего загружено моделей: {len(MODELS)}")
print(f"Классы: {CLASS_NAMES}\n")


# ──────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────

def validate_upload(file: UploadFile, contents: bytes) -> None:
    """Проверяет тип и размер загруженного файла."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Неподдерживаемый формат файла: {file.content_type}. "
                f"Допустимые форматы: JPEG, PNG, WEBP."
            ),
        )

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Файл слишком большой: {size_mb:.1f} MB. "
                f"Максимальный размер: {MAX_FILE_SIZE_MB} MB."
            ),
        )


def run_inference(model_id: str, image: Image.Image, conf: float, iou: float):
    """Запускает инференс выбранной модели и возвращает результаты."""
    entry = MODELS[model_id]
    model = entry["model"]
    imgsz = entry["imgsz"]

    start = time.time()
    results = model.predict(
        image,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    elapsed_ms = (time.time() - start) * 1000

    detections = []
    r = results[0]

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_id in zip(boxes, confs, cls_ids):
            class_name = (
                CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            )
            detections.append(
                {
                    "class": class_name,
                    "class_id": int(cls_id),
                    "confidence": round(float(score), 4),
                    "bbox": [round(float(coord), 2) for coord in box],
                }
            )

    return detections, elapsed_ms


# ──────────────────────────────────────────────────────────────────────────
# Эндпоинты
# ──────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Проверка что сервис жив."""
    return {
        "status": "ok",
        "service": "VisDrone Detection API",
        "models_loaded": list(MODELS.keys()),
    }


@app.get("/health")
def health_check():
    """Простой healthcheck для мониторинга."""
    return {"status": "healthy"}


@app.get("/models")
def list_models():
    """
    Возвращает список доступных моделей для фронтенда:
    id, человекочитаемое имя, описание, mAP, время инференса.
    """
    return {"models": [m["meta"] for m in MODELS.values()]}


@app.get("/classes")
def list_classes():
    """Возвращает список классов, которые умеет находить модель."""
    return {"classes": CLASS_NAMES}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = "fast",
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
):
    """
    Принимает изображение, прогоняет через выбранную модель,
    возвращает список найденных объектов с координатами боксов.

    Параметры:
    - file: изображение (jpg/png/webp)
    - model_id: 'fast' или 'accurate' (см. /models)
    - conf: порог уверенности (по умолчанию 0.25)
    - iou: порог IoU для NMS (по умолчанию 0.7)
    """
    if model_id not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Неизвестная модель '{model_id}'. "
                f"Доступные модели: {list(MODELS.keys())}"
            ),
        )

    contents = await file.read()
    validate_upload(file, contents)

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Не удалось прочитать изображение. Файл повреждён или имеет неподдерживаемый формат.",
        )

    detections, elapsed_ms = run_inference(model_id, image, conf, iou)

    return JSONResponse(
        {
            "model_used": model_id,
            "model_name": MODELS[model_id]["meta"]["name"],
            "image_size": list(image.size),
            "inference_ms": round(elapsed_ms, 1),
            "detections_count": len(detections),
            "detections": detections,
        }
    )


# ──────────────────────────────────────────────────────────────────────────
# Запуск напрямую: python main.py
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)