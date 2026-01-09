# Deep Learning Engineer - 2025"

| Название проекта | Описание | Библиотеки |
| ------------- | ------------- | ------------- |
| Модуль Advanced [Object Detection: FCOS vs YOLO (Minecraft)](./) | **Детекция объектов на кастомном датасете (COCO)**: подготовка датасета (разметка → COCO), обучение и сравнение 2 подходов детекции — **FCOS (MMDetection)** и **YOLO (Ultralytics)**. Проведена оценка качества (**mAP, mAP@50, Precision, Recall, F1**) и скорости (**FPS/время инференса**) на изображениях/видео. Сформированы артефакты: графики обучения, сравнение моделей, таблица метрик и PDF-отчёт. | python, pytorch, mmdetection, mmengine, ultralytics, opencv, numpy, pandas, matplotlib |

---

## Проект: Object Detection (FCOS vs YOLO)

### Цель
Построить пайплайн детекции объектов на изображениях и видео и сравнить две модели:
- **FCOS** (one-stage anchor-free detector) на базе **MMDetection**
- **YOLO** (Ultralytics) как быстрый baseline

### Данные
- Кастомный датасет в стиле **Minecraft**
- Формат аннотаций: **COCO**
- Разбиение: `train / val / test`

Структура (примерно):
- `dataset/minecraft/...` — изображения и аннотации
- `dataset/minecraft/data_coco.yaml` — конфиг для YOLO
- `annotations/*.json` — COCO аннотации для MMDetection

### Что сделано
- Подготовка датасета и приведение аннотаций к формату **COCO**
- Настройка конфигов **MMDetection** под FCOS (кол-во классов, метаинформация, threshold’ы)
- Обучение **FCOS** и **YOLO** + контроль переобучения (patience/early stopping, lr schedule и т.д.)
- Сравнение моделей по качеству и скорости:
  - **mAP, mAP@50**
  - **Precision, Recall, F1-score**
  - **FPS / latency**
- Примеры инференса на изображениях и видео
- Автоматическая генерация `artifacts/report.pdf`

### Основные артефакты
- `artifacts/report.pdf` — итоговый отчёт (метрики + скорость + примеры + графики)
- `artifacts/metrics/metrics_comparison.csv` — таблица сравнения метрик
- `artifacts/inference/...` — визуализации детекций (картинки/кадры)
- `artifacts/yolo_training.jpg`, `artifacts/fcos_training.jpg` — графики обучения
- `artifacts/inference/training_comparetion_pic.jpg` — сравнительная визуализация

## Структура репозитория

```text
DL_EngineerVisionPjct01/
├── notebook02.ipynb                                    #  основной ноутбук проекта (обучение/оценка/отчёт)
├── pdf_reports.py                                      #  создание pdf отчета
├── prepost.py                                          #  предобработка
├── datasetconverter.py                                 #  импортируем  конвертор   VOC XML  в COCO JSON
├── metrics_calc.py                                     #  расчет метрик
├── video_processing.py                                 #  обработка и создание видео
├── configs/
│   └── fcos/
│       └── fcos_minecraft02.py                         #  файл настроек для fcos модели
├── dataset/
│   └── minecraft/
│       ├── data_coco.yaml                              # конфиг датасета для Ultralytics (YOLO)
│       ├── video.mp4                                   # видео для анализа
│       ├── labels.txt                                  # метки классов
│       ├── annotations/                                # COCO json (train/val/test)
│           └── _test_annotations.coco.json             # аннотации для test
│           └── _valid_annotations.coco.json            # аннотации для valid
│           └── _train_annotations.coco.json            # аннотации для train
│       ├── test/                                       # dataset для test
│           └── images                                  # изображения
│           └── labels                                  # метки
│       ├── train/                                      # dataset для test
│           └── images                                  # изображения
│           └── labels                                  # метки
│       ├── valid/                                      # dataset для test
│           └── images                                  # изображения
│           └── labels                                  # метки



│       └── (images/ ...)             # картинки обычно НЕ коммитятся (или через Git LFS)

├── configs/
│   └── fcos_*.py                     # конфиги MMDetection под FCOS (датасет, классы, hooks)



├── dataset/
│   └── minecraft/
│       ├── data_coco.yaml            # конфиг датасета для Ultralytics (YOLO)
│       ├── annotations/              # COCO json (train/val/test) — если небольшой
│       └── (images/ ...)             # картинки обычно НЕ коммитятся (или через Git LFS)
├── artifacts/
│   ├── report.pdf                    # итоговый PDF отчёт
│   ├── metrics/
│   │   └── metrics_comparison.csv    # сравнение метрик моделей (качество/скорость)
│   ├── inference/
│   │   ├── training_comparetion_pic.jpg  # сравнительная картинка обучения/результатов
│   │   └── ...                       # примеры детекций (картинки/кадры)
│   └── yolo/                         # результаты тренировок YOLO (логи/папки runs)
└── README.md                         # описание проекта и как запускать

