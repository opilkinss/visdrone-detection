# Детекция объектов на снимках с дрона (VisDrone)

Веб-сервис для детекции объектов (пешеходы, машины, велосипеды и т.д.) на аэрофотоснимках с дрона. Реализован на YOLO26l, обучен на датасете VisDrone2019-DET.

## Демо

[Открыть приложение](https://visdrone-detection-fpfeqnau4upyhawkmgtnsx.streamlit.app)

## Возможности

- Загрузка собственного изображения или выбор случайного из тестовой выборки
- Выбор между двумя моделями: быстрая ("Поскорее") и точная ("Поточнее")
- Визуализация найденных объектов с bounding box'ами и подписями классов
- Настраиваемый порог уверенности (confidence threshold)

## Архитектура

```
visdrone-project/
├── backend/          FastAPI сервис, обслуживает обе модели
│   ├── main.py
│   ├── models/        веса моделей (.pt, через Git LFS)
│   ├── models_config.json
│   └── requirements.txt
├── frontend/          Gradio веб-интерфейс
│   ├── app.py
│   ├── test_samples/  примеры изображений для демо
│   └── requirements.txt
└── README.md
```

## Метрики моделей

| Модель | Разрешение | mAP@0.5 | Среднее время инференса |
|---|---|---|---|
| Поскорее (fast) | 640px | 0.44 | ~8 мс |
| Поточнее (accurate) | 1280px | 0.588 | ~25 мс |

Обучение: YOLO26l, двухфазный fine-tuning (заморозка backbone → полная разморозка с маленьким lr), датасет VisDrone2019-DET (6471 train / 548 val / 1610 test-dev изображений).

## Классы детекции

pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

## Системные требования

- Python 3.10–3.12 (3.13 может вызывать проблемы со сборкой numpy)
- RAM: минимум 4 GB, рекомендуется 8 GB для модели на 1280px
- Диск: ~500 MB (веса моделей + зависимости)
- CPU достаточно, GPU не обязателен
- ОС: Linux / macOS / Windows

## Установка

```bash
git clone https://github.com/<твой_username>/visdrone-detection.git
cd visdrone-detection

# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
pip install -r requirements.txt
```

## Запуск

В двух отдельных терминалах:

```bash
# Терминал 1 — backend
cd backend
uvicorn main:app --port 8000

# Терминал 2 — frontend
cd frontend
python app.py
```

Интерфейс откроется на `http://localhost:7860`.

## API

| Метод | Путь | Описание |
|---|---|---|
| GET | `/` | Проверка статуса сервиса |
| GET | `/health` | Healthcheck |
| GET | `/models` | Список доступных моделей |
| GET | `/classes` | Список классов детекции |
| POST | `/predict` | Детекция на загруженном изображении |

## Лицензия и данные

Датасет [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) используется в исследовательских/образовательных целях.
