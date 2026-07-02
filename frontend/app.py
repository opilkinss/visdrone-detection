"""
Streamlit frontend для сервиса детекции объектов на снимках с дрона (VisDrone).
Общается с FastAPI backend по HTTP.
"""

import glob
import io
import os
import random

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "http://localhost:8000")

TEST_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_samples"
)

# Палитра
PERIWINKLE = "#B9C7F8"
ZAFFRE     = "#0D21A5"
CITRON     = "#D1D067"

CLASS_COLORS = [
    "#E24B4A", "#378ADD", "#639922", "#BA7517", "#7F77DD",
    "#D85A30", "#D4537E", "#1D9E75", "#888780", "#26215C",
]

# ──────────────────────────────────────────────────────────────────────────
# Настройка страницы и стили
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VisDrone Object Detection",
    page_icon="🛸",
    layout="wide",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, 'SF Pro Display', sans-serif !important;
}}

.stApp {{
    background-color: {PERIWINKLE};
}}

h1, h2, h3 {{
    color: {ZAFFRE} !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}}

p, label, .stMarkdown {{
    color: {ZAFFRE} !important;
}}

.stButton > button {{
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    border: none !important;
    width: 100% !important;
    padding: 10px 20px !important;
    transition: opacity 0.2s !important;
}}

.stButton > button:hover {{
    opacity: 0.85 !important;
}}

div[data-testid="column"]:nth-child(1) .stButton > button {{
    background-color: {CITRON} !important;
    color: {ZAFFRE} !important;
}}

div[data-testid="column"]:nth-child(2) .stButton > button {{
    background-color: {ZAFFRE} !important;
    color: white !important;
}}

.detect-btn > button {{
    background-color: {ZAFFRE} !important;
    color: white !important;
    font-size: 16px !important;
    padding: 14px 20px !important;
}}

div[data-testid="stRadio"] label {{
    color: {ZAFFRE} !important;
    font-weight: 500 !important;
}}

div[data-testid="stSlider"] label {{
    color: {ZAFFRE} !important;
    font-weight: 500 !important;
}}

.stats-box {{
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    color: {ZAFFRE};
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.8;
    margin-top: 12px;
}}

.error-box {{
    background: #fff3f3;
    border-left: 4px solid #E24B4A;
    border-radius: 8px;
    padding: 12px 16px;
    color: #c0392b;
    font-size: 14px;
    margin-bottom: 16px;
}}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_models():
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        resp.raise_for_status()
        return resp.json()["models"], None
    except Exception as e:
        return [], str(e)


def list_test_images():
    if not os.path.isdir(TEST_IMAGES_DIR):
        return []
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext)))
    return files


def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
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
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_h = bbox[3] - bbox[1]
        text_w = bbox[2] - bbox[0]
        draw.rectangle(
            [x1, max(0, y1 - text_h - 6), x1 + text_w + 8, y1],
            fill=color,
        )
        draw.text((x1 + 4, max(0, y1 - text_h - 5)), label, fill="white", font=font)

    return draw_img


def format_stats(result: dict) -> str:
    counts = {}
    for det in result["detections"]:
        counts[det["class"]] = counts.get(det["class"], 0) + 1

    lines = [
        f"<b>Модель:</b> {result['model_name']}",
        f"<b>Размер изображения:</b> {result['image_size'][0]}×{result['image_size'][1]} px",
        f"<b>Время инференса:</b> {result['inference_ms']} мс",
        f"<b>Найдено объектов:</b> {result['detections_count']}",
    ]
    if counts:
        lines.append("<br><b>По классам:</b>")
        for cls_name, count in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"&nbsp;&nbsp;{cls_name}: {count}")

    return "<br>".join(lines)


def run_detection(image: Image.Image, model_id: str, conf: float):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    resp = requests.post(
        f"{API_URL}/predict",
        files={"file": ("image.jpg", buf, "image/jpeg")},
        params={"model_id": model_id, "conf": conf},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────────────────────────────────────────────────────
# Инициализация session_state
# ──────────────────────────────────────────────────────────────────────────

if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "result_image" not in st.session_state:
    st.session_state.result_image = None
if "result_stats" not in st.session_state:
    st.session_state.result_stats = None

# ──────────────────────────────────────────────────────────────────────────
# Интерфейс
# ──────────────────────────────────────────────────────────────────────────

st.title("Детекция объектов на снимках с дрона")
st.markdown(
    "Загрузите своё изображение или возьмите случайное из тестовой выборки "
    " После выберите модель и запустите детекцию."
)

# Проверяем доступность backend
models_list, conn_error = fetch_models()

if conn_error:
    st.markdown(
        f'<div class="error-box">Backend недоступен: {conn_error}<br>'
        f'Проверьте, что FastAPI сервер запущен по адресу {API_URL}</div>',
        unsafe_allow_html=True,
    )

# Строим словарь имя → id
RENAME = {"fast": "Поскорее", "accurate": "Поточнее"}
models_map = {
    RENAME.get(m["id"], m["name"]): m["id"]
    for m in models_list
} if models_list else {"Поскорее": "fast", "Поточнее": "accurate"}

# ── Основной layout ────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    # Кнопки источника изображения
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Случайное из теста"):
            test_images = list_test_images()
            if test_images:
                path = random.choice(test_images)
                st.session_state.current_image = Image.open(path).convert("RGB")
                st.session_state.result_image = None
                st.session_state.result_stats = None
            else:
                st.warning("Папка test_samples пуста или не найдена.")

    with btn_col2:
        uploaded = st.file_uploader(
            "Загрузить файл",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.session_state.current_image = Image.open(uploaded).convert("RGB")
            st.session_state.result_image = None
            st.session_state.result_stats = None

    # Превью изображения
    if st.session_state.current_image:
        st.image(st.session_state.current_image, use_container_width=True)
    else:
        st.markdown(
            '<div style="border:2px dashed #0D21A5; border-radius:12px; '
            'height:280px; display:flex; align-items:center; '
            'justify-content:center; color:#0D21A5; opacity:0.5;">'
            'Изображение появится здесь</div>',
            unsafe_allow_html=True,
        )

    # Выбор модели
    model_name = st.radio(
        "Модель",
        options=list(models_map.keys()),
        horizontal=True,
    )

    # Слайдер confidence
    conf_threshold = st.slider(
        "Порог уверенности (confidence)",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    # Кнопка детекции
    st.markdown('<div class="detect-btn">', unsafe_allow_html=True)
    detect_clicked = st.button("Детектировать")
    st.markdown('</div>', unsafe_allow_html=True)

    if detect_clicked:
        if st.session_state.current_image is None:
            st.warning("Сначала загрузите изображение или выберите случайное.")
        elif conn_error:
            st.error("Backend недоступен. Проверьте соединение.")
        else:
            with st.spinner("Выполняется детекция..."):
                try:
                    model_id = models_map[model_name]
                    result = run_detection(
                        st.session_state.current_image,
                        model_id,
                        conf_threshold,
                    )
                    st.session_state.result_image = draw_boxes(
                        st.session_state.current_image,
                        result["detections"],
                    )
                    st.session_state.result_stats = format_stats(result)
                except Exception as e:
                    st.error(f"Ошибка при детекции: {e}")

with right_col:
    if st.session_state.result_image:
        st.image(st.session_state.result_image, use_container_width=True)
    else:
        st.markdown(
            '<div style="border:2px dashed #0D21A5; border-radius:12px; '
            'height:280px; display:flex; align-items:center; '
            'justify-content:center; color:#0D21A5; opacity:0.5;">'
            'Результат с боксами появится здесь</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.result_stats:
        st.markdown(
            f'<div class="stats-box">{st.session_state.result_stats}</div>',
            unsafe_allow_html=True,
        )

# Footer
st.markdown("---")
st.markdown(
    f'<p style="font-size:12px; color:{ZAFFRE}; opacity:0.7;">'
    "Классы: pedestrian, people, bicycle, car, van, truck, "
    "tricycle, awning-tricycle, bus, motor</p>",
    unsafe_allow_html=True,
)