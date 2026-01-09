import yaml
import cv2
import glob, os
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
import mmcv
import numpy as np
from pathlib import Path
import shutil
import time                                                                 # для FPS
## проверка и визуализация по датасетам


def verify_and_visualize(data_labels_path: str, dataset: str):
    # Чтение и парсинг labels.txt
    try:
        with open(data_labels_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print('alarm')
        return

    dataset_root_path_rel = "dataset/minecraft/"
    images_notation_rel_path = dataset_root_path_rel + dataset
    # проверка на пропущенные файлы разметки
    images_without_labels = []
    labels_without_images = []


    image_files = glob.glob(images_notation_rel_path + '/*.jpg')
    label_files = glob.glob(images_notation_rel_path+ '/*.xml')

    image_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in image_files}
    label_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in label_files}

    # Получаем множества ключей (имён файлов) для быстрой работы
    image_keys = set(image_stems.keys())
    label_keys = set(label_stems.keys())

    # Находим имена файлов, которые есть в одном множестве, но отсутствуют в другом
    images_without_labels_stems = image_keys - label_keys
    labels_without_images_stems = label_keys - image_keys

    # Формируем списки полных путей для файлов без пары
    missing_img = [image_stems[stem] for stem in images_without_labels_stems]
    missing_lbl = [label_stems[stem] for stem in labels_without_images_stems]

    # Формируем отчёт
    report = []
    total_images = len(image_files)
    total_labels = len(label_files)
    
    if not missing_img and not missing_lbl:
        report.append("Все файлы имеют соответствующие пары:")
        report.append(f"  Изображений: {total_images}")
        report.append(f"  Меток: {total_labels}")
    else:
        if missing_img:
            report.append("Изображения без соответствующих меток:")
            for img in missing_img:
                report.append(f"  - {os.path.basename(img)}")
            report.append(f"Всего: {len(missing_img)} изображений")
        
        if missing_lbl:
            report.append("\nМетки без соответствующих изображений:")
            for lbl in missing_lbl:
                report.append(f"  - {os.path.basename(lbl)}")
            report.append(f"Всего: {len(missing_lbl)} меток")
        
        report.append("\nИтоговая статистика:")
        report.append(f"  Всего изображений: {total_images}")
        report.append(f"  Всего меток: {total_labels}")
        report.append(f"  Полных пар: {min(total_images, total_labels) - max(len(missing_img), len(missing_lbl))}")

    print("\n".join(report))

    # Выбор случайного изображения и его разметки
    all_images = [f for f in os.listdir(images_notation_rel_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        return
        
    random_image_name = random.choice(all_images)
    image_path = os.path.join(images_notation_rel_path, random_image_name)

    print(random_image_name)

    # Сформируйте путь к соответствующему файлу разметки (`.xml`).
    label_name = os.path.splitext(random_image_name)[0] + '.xml'
    label_path = os.path.join(images_notation_rel_path, label_name)

    # Визуализация
    plt.figure(figsize=(7, 12))
    image = cv2.imread(image_path)
    
    if not os.path.exists(label_path):
        print("Для этого изображения нет файла разметки")
    else:
         # Разбор XML
        tree = ET.parse(label_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            bbox = obj.find("bndbox")
            x_min = int(bbox.find("xmin").text)
            y_min = int(bbox.find("ymin").text)
            x_max = int(bbox.find("xmax").text)
            y_max = int(bbox.find("ymax").text)
            # Рисуем рамку
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Рисуем название класса
            cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Отображаем результат
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Верификация разметки: {dataset}")
    plt.axis('off')
    plt.show()

   
## распределение классов

def class_distribution(data_labels_path: str, dataset: str):

    dataset_root_path_rel = "dataset/minecraft/"
    images_notation_rel_path = dataset_root_path_rel + dataset
    label_files = glob.glob(images_notation_rel_path+ '/*.xml')
    # дисбаланс классов
    class_counter = Counter()
  
    for file_path in label_files:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            class_id = obj.find("name").text
            class_counter[class_id] += 1  

    sorted_class_ids = sorted(class_counter.keys())
  

    counts = [class_counter[i] for i in sorted_class_ids]
    counts_norm = np.array(counts) / np.sum(counts)

    plt.figure(figsize=(8, 5))

    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=sorted_class_ids, y=counts_norm, hue=sorted_class_ids,  palette="viridis", legend=False)
    ax.set_title('Распределение классов в обучающем датасете', fontsize=18)

    ax.set_xlabel('Классы', fontsize=12)
    ax.set_ylabel('Количество экземпляров', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('artifacts/class_distribution.png')
    plt.show()


## отрисовка результатов для fcos

def draw_fcos_img(model, result, img_path, out_path, score_thr,
                          box_color=(0, 255, 0),
                          text_color=(0, 255, 0),
                          thickness=2):
    """
    Рисует bounding boxes FCOS и сохраняет итоговое изображение.

    Args:
        model: FCOS модель (init_detector)
        result: результат model.test (DataSample)
        img_path: путь к источнику изображения
        out_path: куда сохранить итоговую картинку
        box_color: цвет прямоугольника (BGR)
        text_color: цвет текста (BGR)
        thickness: толщина линии
    """
    
    # --- загрузка изображения ---
    img = mmcv.imread(img_path)
    
    # -- извлечение предсказаний ---
    pred = result.pred_instances

    # --- фильтрация по score ---
    scores_cpu = pred.scores.cpu().numpy()
    mask = scores_cpu > score_thr

    bboxes = pred.bboxes[mask].cpu().numpy()
    labels = pred.labels[mask].cpu().numpy()
    scores = scores_cpu[mask]
    
    class_names = model.dataset_meta["classes"]

    # --- рисуем ---
    for box, label, score in zip(bboxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

        text = f"{class_names[label]} {score:.2f}"
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)


    # --- сохранить ---
    mmcv.imwrite(img, out_path)
    print(f"[OK] Saved FCOS result to: {out_path}")

    return out_path


def load_classes(classes_file: str):
    """Считать список классов из файла, по одному в строке."""
    classes_path = Path(classes_file)
    if not classes_path.exists():
        raise FileNotFoundError(f"Файл classes.txt не найден: {classes_file}")
    return classes_path.read_text(encoding="utf-8").splitlines()


## взято отсюдова
## https://medium.com/internet-of-technology/convert-pascal-voc-to-yolo-format-b7672bcf0cb3

"""
convert_voc_to_yolo.py

Конвертер VOC XML → YOLO TXT, который:
- создаёт подпапку labels/
- конвертирует все .xml → .txt
- читает классы из файла labels.txt
- подходит для вызова из Jupyter Notebook
"""

def load_classes(classes_file: str):
    """Считать список классов из labels.txt, очищая пробелы и пустые строки."""
    classes_path = Path(classes_file)

    if not classes_path.exists():
        raise FileNotFoundError(f"Файл classes (labels.txt) не найден: {classes_file}")

    raw = classes_path.read_text(encoding="utf-8").splitlines()
    classes = [c.strip() for c in raw if c.strip() != ""]

    print(f"[INFO] Загружено классов: {len(classes)}")
    return classes


def convert_bbox(size, box):
    """Преобразование bbox VOC в YOLO формат."""
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box

    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    return x_center, y_center, width, height


def convert_single_xml(xml_path: Path, txt_path: Path, classes: list):
    """Конвертация одного XML → TXT."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] Ошибка парсинга {xml_path}: {e}")
        return

    size_el = root.find("size")
    if size_el is None:
        print(f"[WARN] Файл {xml_path} не содержит тег <size>")
        return

    img_w = int(size_el.find("width").text)
    img_h = int(size_el.find("height").text)

    with txt_path.open("w", encoding="utf-8") as f:
        for obj in root.iter("object"):
            class_name = obj.find("name").text.strip()

            if class_name not in classes:
                print(f"[WARN] Класс '{class_name}' из {xml_path} отсутствует в labels.txt")
                continue

            class_id = classes.index(class_name)

            box = obj.find("bndbox")
            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)

            x_c, y_c, w, h = convert_bbox((img_w, img_h), (xmin, ymin, xmax, ymax))
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def convert_voc_folder(folder_path: str, classes_file: str):
    """
    Конвертация всех XML в одной папке.
    Создаёт папку labels/.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder_path}")

    # грузим классы
    classes = load_classes(classes_file)

  
    # создаём images/ и labels/ 
    images_dir = folder / "images"
    labels_dir = folder / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # копируем все изображения в images/
    img_exts = (".jpg", ".jpeg", ".png")
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in img_exts]

    print(f"[INFO] Найдено изображений: {len(image_files)}")

    for img_path in image_files:
        shutil.copy(img_path, images_dir / img_path.name)

    # ищем XML файлы
    xml_files = list(folder.glob("*.xml"))
    print(f"[INFO] Найдено XML файлов в {folder}: {len(xml_files)}")

    for xml_path in xml_files:
        txt_path = labels_dir / (xml_path.stem + ".txt")
        convert_single_xml(xml_path, txt_path, classes)

    print(f"[DONE] Конвертированы TXT → {labels_dir}")
    print(f"[DONE] Изображения скопированы → {images_dir}")



def measure_fps(models_dict, image_input, warmup=10, num_runs=100):

    results = {}
    # -------- WARMUP stage --------    
    for model_name, model in models_dict.items():
        # Обрабатываем модели
        if "YOLO" in model_name.upper():                         # Для YOLO
            for _ in range(warmup):
                model.predict(
                    source=image_input,
                    verbose=False,
                    save=False,
                    show=False
                )
        else:                                            # Для FCOS
            for _ in range(warmup):
               model(
                    image_input,
                    show=False,
                    print_result=False,
                    no_save_pred=True,
                );



    # -------- TIMING stage --------
    for model_name, model in models_dict.items():
        start_time = time.time()
        # Обрабатываем модели
        if "YOLO" in model_name.upper():                         # Для YOLO
            for _ in range(num_runs):
                model.predict(
                    source=image_input,
                    verbose=False,
                    save=False,
                    show=False
                )
        else:                                            # Для FCOS
            for _ in range(num_runs):
               model(
                    image_input,
                    show=False,
                    print_result=False,
                    no_save_pred=True,
                    
                )


        end_time = time.time()
        fps = num_runs / (end_time - start_time)
        results[model_name] = fps

    # Вывод результатов
    print("\n" + "="*20 + " Результаты измерения End-to-End FPS " + "="*20)
    for name, fps in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{name:<15}: {fps:.2f} FPS")

    return results 



    

if __name__ == '__main__':
    # Укажите путь к вашему `data.yaml`, который вы создали в Части 1.
    path_to_labels= "datasets/minecraft/labels.txt"
    dataset = 'train'
    verify_and_visualize(path_to_labels, dataset) 