# Deep Learning Engineer - 2025"

| Название проекта | Описание | Библиотеки |
| ------------- | ------------- | ------------- |
| Модуль 1 [M01P01-Borrowers](https://github.com/MalastowskiNS/-YanXPraKPjcts/blob/main/M01P01-Borrowers/borrow_it.ipynb) | **Исследование надежности заемщиков**: Для кредитного отдел банка нужно разобраться, влияет ли семейное положение и количество детей клиента на факт погашения кредита в срок. Входные данные от банка — статистика о платёжеспособности клиентов. Результаты исследования необходимы для построения модели кредитного скоринга. | pandas, seaborn, numpy, matplotlib |
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

### Как запустить
1) Установить зависимости (примерно):
```bash
pip install -r requirements.txt
