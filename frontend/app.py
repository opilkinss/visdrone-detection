"""
Gradio frontend для сервиса детекции объектов на снимках с дрона (VisDrone).
Общается с FastAPI backend по HTTP.
"""

import glob
import io
import os
import random

import gradio as gr
import requests
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Папка с несколькими примерами для кнопки "случайное изображение"
TEST_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_samples")

BOX_COLOR = "#E24B4A"
TEXT_COLOR = "#FFFFFF"

# Палитра — разный цвет на класс, чтобы боксы было легче различать
CLASS_COLORS = [
    "#E24B4A", "#378ADD", "#639922", "#BA7517", "#7F77DD",
    "#D85A30", "#D4537E", "#1D9E75", "#888780", "#26215C",
]


# ──────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────

def fetch_models():
    """Получает список моделей с backend. При ошибке возвращает понятное сообщение."""
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()["models"]
        name_to_id = {m["name"]: m["id"] for m in data}
        info_by_id = {m["id"]: m for m in data}
        return name_to_id, info_by_id, None
    except requests.exceptions.RequestException as e:
        return {}, {}, f"Не удалось подключиться к backend по адресу {API_URL}: {e}"


MODELS_MAP, MODELS_INFO, CONNECTION_ERROR = fetch_models()

if CONNECTION_ERROR:
    # Заглушка, чтобы интерфейс не падал, если backend ещё не запущен
    MODELS_MAP = {"Поскорее (недоступно)": "fast", "Поточнее (недоступно)": "accurate"}
else:
    # Переименовываем модели в дружелюбные названия для интерфейса,
    # сохраняя при этом исходные id для общения с backend
    RENAME = {"fast": "Поскорее", "accurate": "Поточнее"}
    renamed_map = {}
    for display_name, model_id in MODELS_MAP.items():
        new_name = RENAME.get(model_id, display_name)
        renamed_map[new_name] = model_id
    MODELS_MAP = renamed_map


def list_test_images():
    if not os.path.isdir(TEST_IMAGES_DIR):
        return []
    patterns = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, p)))
    return files


TEST_IMAGES = list_test_images()


def load_random_test_image():
    """Возвращает случайное изображение из тестовой папки, либо None с сообщением."""
    if not TEST_IMAGES:
        gr.Warning("Папка test_samples пуста или не найдена.")
        return None
    path = random.choice(TEST_IMAGES)
    return Image.open(path).convert("RGB")


def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Рисует bounding box'ы с подписями поверх изображения."""
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    class_to_color = {}

    for det in detections:
        cls_name = det["class"]
        if cls_name not in class_to_color:
            idx = len(class_to_color) % len(CLASS_COLORS)
            class_to_color[cls_name] = CLASS_COLORS[idx]
        color = class_to_color[cls_name]

        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"{cls_name} {det['confidence']:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle(
            [x1, max(0, y1 - text_h - 6), x1 + (text_bbox[2] - text_bbox[0]) + 8, y1],
            fill=color,
        )
        draw.text((x1 + 4, max(0, y1 - text_h - 5)), label, fill=TEXT_COLOR, font=font)

    return draw_img


def format_stats(result: dict) -> str:
    """Формирует читаемый текстовый отчёт по результату детекции."""
    counts = {}
    for det in result["detections"]:
        counts[det["class"]] = counts.get(det["class"], 0) + 1

    lines = [
        f"Модель: {result['model_name']}",
        f"Размер изображения: {result['image_size'][0]}x{result['image_size'][1]}",
        f"Время инференса: {result['inference_ms']} мс",
        f"Всего найдено объектов: {result['detections_count']}",
    ]

    if counts:
        lines.append("")
        lines.append("По классам:")
        for cls_name, count in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {cls_name}: {count}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Основная функция детекции
# ──────────────────────────────────────────────────────────────────────────

def detect(image, model_name, conf_threshold):
    if image is None:
        return None, "Сначала загрузите изображение или нажмите «Случайное из теста»."

    if model_name not in MODELS_MAP:
        return None, "Выбранная модель недоступна."

    model_id = MODELS_MAP[model_name]

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    try:
        resp = requests.post(
            f"{API_URL}/predict",
            files={"file": ("image.jpg", buf, "image/jpeg")},
            params={"model_id": model_id, "conf": conf_threshold},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return None, f"Ошибка запроса к backend: {e}"

    result = resp.json()
    annotated = draw_boxes(image, result["detections"])
    stats = format_stats(result)

    return annotated, stats


# ──────────────────────────────────────────────────────────────────────────
# Интерфейс
# ──────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root, .gradio-container {
    --font: 'Inter', -apple-system, 'SF Pro Display', 'SF Pro Text',
            'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    --periwinkle: #B9C7F8;
    --zaffre: #0D21A5;
    --citron: #D1D067;
}

.gradio-container, .gradio-container * {
    font-family: var(--font) !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    background: var(--periwinkle) !important;
}

body, .gradio-container {
    background: var(--periwinkle) !important;
}

h1 {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    font-size: 28px !important;
    color: var(--zaffre) !important;
}

h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    color: var(--zaffre) !important;
}

p {
    color: var(--zaffre) !important;
}

.gr-button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    letter-spacing: -0.01em !important;
}

.gr-button-primary {
    background: var(--zaffre) !important;
    border: none !important;
    color: #ffffff !important;
}

.gr-button-primary:hover {
    background: #091a85 !important;
}

.gr-button-secondary {
    background: var(--citron) !important;
    border: none !important;
    color: var(--zaffre) !important;
}

.gr-button-secondary:hover {
    background: #bdbc52 !important;
}

.gr-box, .gr-panel, .gr-form {
    border-radius: 12px !important;
    background: #ffffff !important;
}

textarea, input[type="text"], input[type="number"] {
    font-family: var(--font) !important;
    font-size: 14px !important;
}

label, .gr-text-input label, span.svelte-1gfkn6j {
    font-weight: 500 !important;
    font-size: 13px !important;
    color: var(--zaffre) !important;
}
"""

THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="yellow",
    neutral_hue="blue",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "monospace"],
).set(
    button_primary_background_fill="#0D21A5",
    button_primary_background_fill_hover="#091a85",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#D1D067",
    button_secondary_background_fill_hover="#bdbc52",
    button_secondary_text_color="#0D21A5",
    block_radius="12px",
    button_large_radius="10px",
    body_background_fill="#B9C7F8",
    block_background_fill="#ffffff",
)

with gr.Blocks(title="VisDrone Object Detection", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.Markdown("# Детекция объектов на снимках с дрона")
    gr.Markdown(
        "Загрузите своё изображение, возьмите случайное из тестовой выборки "
        "или узнайте какой Вы участник дорожного движения и сделайте фотографию "
        "с веб-камеры. После выберите модель и запустите детекцию."
    )

    if CONNECTION_ERROR:
        gr.Markdown(
            f"**Backend недоступен:** {CONNECTION_ERROR}\n\n"
            f"Проверьте, что FastAPI сервер запущен и адрес `API_URL` указан верно."
        )

    with gr.Row():
        with gr.Column():
            inp_image = gr.Image(type="pil", label="Изображение")

            btn_random = gr.Button("Случайное изображение из теста", variant="secondary")

            inp_model = gr.Radio(
                choices=list(MODELS_MAP.keys()),
                value=list(MODELS_MAP.keys())[0] if MODELS_MAP else None,
                label="Модель",
            )

            inp_conf = gr.Slider(
                minimum=0.05,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Порог уверенности (confidence)",
            )

            btn_detect = gr.Button("Детектировать", variant="primary")

        with gr.Column():
            out_image = gr.Image(label="Результат")
            out_text = gr.Textbox(label="Статистика", lines=8)

    btn_random.click(fn=load_random_test_image, outputs=inp_image)
    btn_detect.click(
        fn=detect,
        inputs=[inp_image, inp_model, inp_conf],
        outputs=[out_image, out_text],
    )

    gr.Markdown(
        f"---\nКлассы детекции: {', '.join(MODELS_INFO.get('fast', {}).get('classes', [])) or 'pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor'}"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)