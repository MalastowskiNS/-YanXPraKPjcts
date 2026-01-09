import os
import glob
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from fpdf import FPDF
from fpdf.enums import XPos, YPos

class PDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use only built-in core fonts:
        # Helvetica / Times / Courier (no Cyrillic support!)
        self.font_family = "Helvetica"

        # Optional: set default page margins
        self.set_auto_page_break(auto=True, margin=15)



    def header(self):
        """Создаёт шапку для каждой страницы"""
        self.set_font(self.font_family, 'B', 15)
        self.cell(0, 10, 'Analitical report', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font(self.font_family, '', 8)
        self.cell(0, 5, f'data of creation: {datetime.date.today().strftime("%d.%m.%Y")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        """Добавляет номера страниц в подвале"""
        self.set_y(-15)
        self.set_font(self.font_family, 'B', 8)
        self.cell(0, 10, f'page {self.page_no()}', border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')

    def chapter_title(self, title, align="L", size=12, style="B"):
        """Создаёт заголовок раздела"""
        self.set_font(self.font_family, style, size)
        self.cell(0, 10, title, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=align)
        self.ln(5)

    def chapter_body(self, body):
        """Добавляет основной текст"""
        self.set_font(self.font_family, '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
    
    def add_image_section(self, title, image_path, stats_text):
        """Добавляет секцию с изображением и статистикой"""
        self.add_page()
        self.chapter_title(title)

        # Центрируем изображение на странице
        image_width = 100
        page_width = self.w - 2 * self.l_margin
        x_position = (page_width - image_width) / 2 + self.l_margin
        self.image(image_path, x=x_position, y=None, w=image_width)
        self.ln(5)
        self.set_font(self.font_family, '', 10) 
        self.multi_cell(0, 5, stats_text) 
    


def generate_project_report (
    dataset_storage  = "dataset/minecraft",                     # исходный датасет
    config_path = "configs/fcos/fcos_minecraft.py",             # конфиг файл
    artifacts_path = "artifacts",                               # папка с результатами
    out_pdf="artifacts/",                                       # папка для записи отчета
    max_examples=4,
    ):
    pdf = PDFReport()
  
    # --- 1. preview - task page ---
    pdf.add_page()
    labels_path = Path(dataset_storage) / "labels.txt"
    with open(labels_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    classes_str = ", ".join(classes)

    # --- считаем количество изображений в train/val/test по структуре ТЗ ---
    # 1) если есть split/images -> считаем там
    # 2) иначе считаем файлы прямо в split/
    train_images = len(os.listdir(Path(dataset_storage) / "train" / "images"))
    valid_images   = len(os.listdir(Path(dataset_storage) / "valid"   / "images"))
    test_images  = len(os.listdir(Path(dataset_storage) / "test"  / "images"))

    # --- остальное (пока константы как на титуле) ---
    test_objects = 351
    epochs = 12
    gpu_name = "RTX 4060 Laptop"
    report_date = "2026-01-08"

    # --- титульная ---
    pdf.chapter_title("Object Detection in Minecraft", align="C", size=22, style="B")
    pdf.chapter_title("Comparison of FCOS (MMDetection) and YOLOv8n (Ultralytics)", align="C", size=11, style="")
    
    summary_text = (
        f"\n"
        f"- Dataset splits: train={train_images} images, valid={valid_images} images, test={test_images} images.\n"
        f"- Test objects: {test_objects} (from YOLO validation log).\n"
        f"- Classes ({len(classes)}): {classes_str}.\n"
        f"- Training: {epochs} epochs (both runs), GPU: {gpu_name} (from logs).\n"
        f"\n"
        f"Date: {report_date}\n"
    )

    pdf.chapter_body(summary_text)


    # --- 2. page - definition ---
    pdf.add_page()
    pdf.chapter_title("2. Project task and notebook structure")

    page2_text = (
        "Project: Scanning a cubic world - object detection of Minecraft characters using FCOS and YOLO.\n\n"

        "Goal\n"
        "Fine-tune two object detectors (FCOS and YOLO) to recognize Minecraft mobs and compare the models "
        "by accuracy, inference speed, and visual quality of predictions.\n\n"

        "Notebook sections (5 parts)\n"
        "1) Data preparation and EDA\n"
        "- Verify dataset structure and annotations consistency.\n"
        "- Compare the number of images and annotations, detect possible issues.\n"
        "- Analyze class distribution and check for imbalance.\n"
        "- Visualize sample images with ground-truth bounding boxes and class labels.\n\n"

        "2) FCOS (MMDetection)\n"
        "- Configure FCOS for the Minecraft classes (metainfo, pipelines, training parameters).\n"
        "- Run inference on a pretrained FCOS model to validate the pipeline.\n"
        "- Fine-tune FCOS and track training dynamics.\n"
        "- Save model checkpoints, logs, and visualizations.\n\n"

        "3) YOLO (Ultralytics)\n"
        "- Prepare YOLO dataset configuration (YAML) for the same classes.\n"
        "- Run inference on a pretrained YOLO model to validate the pipeline.\n"
        "- Fine-tune YOLO and monitor training metrics.\n"
        "- Save training results and validation outputs.\n\n"

        "4) Inference: images and video\n"
        "- Run inference on test images for both models.\n"
        "- Save qualitative examples with bounding boxes, confidence scores, and class names.\n"
        "- Run video inference and export annotated videos for FCOS and YOLO.\n\n"

        "5) Metrics comparison and conclusions\n"
        "- Compute standard metrics on the test set: Precision, Recall, F1-score, mAP, mAP@50.\n"
        "- Measure inference speed: FPS and/or average inference time.\n"
        "- Build comparison plots and summarize results.\n"
        "- Provide final conclusions: which model is better for accuracy, which is faster, and typical failure cases.\n"
    )

    pdf.chapter_body(page2_text)

    # ---------- Page 3: Section 1 (EDA) - sample image ----------
    pdf.add_page()
    pdf.chapter_title("3. Section 1 - Data and EDA")

    section1_work = (
        "Work done:\n"
        "- Verified dataset split structure (train/val/test) and file integrity.\n"
        "- Checked that annotations match the images.\n"
        "- Visualized one test example with bounding boxes and class labels.\n"
    )
    pdf.chapter_body(section1_work)

    pdf.image(r"artifacts\test_img.jpg", x=20, w=170)


    # ---------- Page 4: Section 1 (EDA) - class distribution ----------
    pdf.add_page()
    pdf.chapter_title("3. Section 1 - Data and EDA (continued)")

    pdf.image(r"artifacts\class_distribution.png", x=20, w=170)
    pdf.ln(5)

    section1_conc = (
        "Conclusions:\n"
        "- The dataset is suitable for training and evaluation.\n"
        "- Class distribution shows imbalance: rare classes may have lower recall.\n"
        "- EDA helps detect annotation issues before training.\n"
    )
    pdf.chapter_body(section1_conc)

    # ---------- Page 5: Pretrained FCOS inference ----------
    pdf.add_page()
    pdf.chapter_title("4. Pretrained models - FCOS (baseline)")

    text_fcos = (
        "Work done:\n"
        "- Loaded a pretrained FCOS checkpoint and ran inference on a test image.\n"
        "- Saved the visualization with predicted bounding boxes, scores, and class labels.\n\n"
    )
    pdf.chapter_body(text_fcos)

    pdf.image(r"artifacts\inference\pretrain\test_pretrained_fcos.jpg", x=20, w=170)
    pdf.ln(5)

    fcos_conc = (
        "Conclusions:\n"
        "- The model detects some objects but is not adapted to the Minecraft domain.\n"
        "- Missed detections and wrong classes are expected before fine-tuning.\n"
        "- This result is a useful baseline to compare with the fine-tuned FCOS model.\n"
    )
    pdf.chapter_body(fcos_conc)


    # ---------- Page 6: Pretrained YOLO inference ----------
    pdf.add_page()
    pdf.chapter_title("4. Pretrained models - YOLO (baseline)")

    text_yolo = (
        "Work done:\n"
        "- Loaded a pretrained YOLO model and ran inference on a test image.\n"
        "- Saved the visualization with predicted bounding boxes, confidence, and class labels.\n\n"
    )
    pdf.chapter_body(text_yolo)

    pdf.image(r"artifacts\inference\pretrain\test_pretrained_yolo.jpg", x=20, w=170)
    pdf.ln(5)

    yolo_conc = (
        "Conclusions:\n"
        "- Pretrained YOLO produces plausible boxes, but domain shift causes false positives/negatives.\n"
        "- Confidence calibration is not reliable for Minecraft classes before training.\n"
        "- This baseline helps evaluate the improvement after fine-tuning.\n"
    )
    pdf.chapter_body(yolo_conc)

# ---------- Section 3: Training dynamics (YOLO + FCOS) ----------

    # Page: YOLO training curves
    pdf.add_page()
    pdf.chapter_title("5. Section 3 - Training dynamics (YOLO)")

    yolo_train_text = (
    "Work done:\n"
    "- Tracked YOLO training and validation curves during fine-tuning.\n"
    "- Training used a warm start: the model had been pretrained for 72 epochs before this short run.\n\n"
    "Training assessment (from curves):\n"
    "- Train losses (box, cls, dfl) decrease steadily across epochs, showing stable optimization.\n"
    "- Validation losses generally go down but are noisy, which is typical for a small/imbalanced dataset.\n"
    "- Recall increases and reaches about 0.33-0.35 by the end of training.\n"
    "- Precision is unstable (spikes and drops), likely due to limited validation data and threshold sensitivity.\n"
    "- mAP@50 grows to about 0.18-0.19 and mAP@50-95 to about 0.12-0.13, then starts to plateau.\n"
)
    pdf.chapter_body(yolo_train_text)

    pdf.image(r"artifacts\yolo_training.jpg", x=20, w=170)
    pdf.ln(5)

    yolo_conc = (
    "Conclusions:\n"
    "- Warm start helps: the model improves quickly and converges within ~10-12 epochs.\n"
    "- The main quality gains happen early; later epochs give smaller improvements.\n"
    "- Precision instability suggests tuning confidence threshold and adding more validation samples could help.\n"
    )
    pdf.chapter_body(yolo_conc)


    # Page: FCOS training curves
    pdf.add_page()
    pdf.chapter_title("5. Section 3 - Training dynamics (FCOS)")

    fcos_train_text = (
    "Work done:\n"
    "- Tracked FCOS training losses during fine-tuning.\n"
    "- Total loss decreases from about 1.9 to about 0.9 over the run, indicating good convergence.\n"
    "- Classification loss drops strongly (about 0.8 to ~0.11), suggesting the classifier adapted well.\n"
    "- BBox regression loss decreases (about 0.53 to ~0.22) with minor noise, then stabilizes.\n"
    "- Centerness loss changes slightly (about 0.61 to ~0.56) and stays stable, which is expected.\n"
    "- After roughly epochs 7-9 the curves mostly flatten, meaning further training gives limited benefit.\n"
    )
    pdf.chapter_body(fcos_train_text)

    pdf.image(r"artifacts\fcos_training.jpg", x=20, w=170)
    pdf.ln(5)

    fcos_conc = (
    "Conclusions:\n"
    "- Fine-tuning after 72-epoch pretraining converges quickly and is stable.\n"
    "- Losses show no divergence; training looks healthy.\n"
    "- Since curves plateau near the end, a short schedule (~12 epochs) is sufficient for this stage.\n"
    )
    pdf.chapter_body(fcos_conc)


    # ---------- Section 4: Comparison (image + metrics table + video notes) ----------

    # --- read metrics (use attached file path OR project path) ---
    metrics_csv_path = 'artifacts/metrics/metrics_comparison.csv'  # file you attached here
    # if you generate on Windows locally, use this instead:
    # metrics_csv_path = r"artifacts\metrics\metrics_comparison.csv"

    metrics_df = pd.read_csv(metrics_csv_path)

    # detect the model-name column (usually "Unnamed: 0")
    name_col = "Unnamed: 0" if "Unnamed: 0" in metrics_df.columns else metrics_df.columns[0]

    # get rows
    yolo_row = metrics_df[metrics_df[name_col].astype(str).str.contains("YOLO", case=False)].iloc[0]
    fcos_row = metrics_df[metrics_df[name_col].astype(str).str.contains("FCOS", case=False)].iloc[0]

    # overall metrics columns in your CSV
    fps_yolo = float(yolo_row["metric"])
    map50_yolo = float(yolo_row["metric.1"])
    map5095_yolo = float(yolo_row["metric.2"])

    fps_fcos = float(fcos_row["metric"])
    map50_fcos = float(fcos_row["metric.1"])
    map5095_fcos = float(fcos_row["metric.2"])

    # macro averages from per-class columns
    prec_cols = [c for c in metrics_df.columns if str(c).startswith("precision")]
    rec_cols  = [c for c in metrics_df.columns if str(c).startswith("recall")]
    f1_cols   = [c for c in metrics_df.columns if str(c).startswith("f1")]

    vals_yolo_p = pd.to_numeric(yolo_row[prec_cols].replace("-", np.nan), errors="coerce")
    vals_yolo_r = pd.to_numeric(yolo_row[rec_cols].replace("-", np.nan), errors="coerce")
    vals_yolo_f = pd.to_numeric(yolo_row[f1_cols].replace("-", np.nan), errors="coerce")

    vals_fcos_p = pd.to_numeric(fcos_row[prec_cols].replace("-", np.nan), errors="coerce")
    vals_fcos_r = pd.to_numeric(fcos_row[rec_cols].replace("-", np.nan), errors="coerce")
    vals_fcos_f = pd.to_numeric(fcos_row[f1_cols].replace("-", np.nan), errors="coerce")

    prec_yolo = float(vals_yolo_p.mean(skipna=True))
    rec_yolo  = float(vals_yolo_r.mean(skipna=True))
    f1_yolo   = float(vals_yolo_f.mean(skipna=True))

    prec_fcos = float(vals_fcos_p.mean(skipna=True))
    rec_fcos  = float(vals_fcos_r.mean(skipna=True))
    f1_fcos   = float(vals_fcos_f.mean(skipna=True))

    speed_ratio = fps_yolo / fps_fcos


    # ---------- Page: qualitative comparison image (fit to page) ----------
    pdf.add_page()
    pdf.chapter_title("Section 4 - Qualitative comparison")

    img_path = r"artifacts\inference\training_comparetion_pic.jpg"
    # on Linux use: "artifacts/inference/training_comparetion_pic.jpg"

    # Fit image to page while keeping aspect ratio
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    max_h = pdf.h - pdf.get_y() - pdf.b_margin

    im = Image.open(img_path)
    w0, h0 = im.size
    scale = min(page_w / w0, max_h / h0)
    w = w0 * scale
    h = h0 * scale
    x = pdf.l_margin + (page_w - w) / 2

    pdf.image(img_path, x=x, y=pdf.get_y(), w=w)
    pdf.ln(5)


    # ---------- Page: metrics table ----------
    pdf.add_page()
    pdf.chapter_title("Section 4 - Metrics comparison (test set)")

    pdf.set_font(pdf.font_family, "B", 10)

    # Table layout
    col_w = [42, 22, 26, 28, 24, 24, 24]  # sum fits ~190mm
    headers = ["Model", "FPS", "mAP@50", "mAP@50-95", "Prec", "Recall", "F1"]

    for i, htxt in enumerate(headers):
        pdf.cell(col_w[i], 8, htxt, border=1, align="C")
    pdf.ln()

    pdf.set_font(pdf.font_family, "", 10)

    rows = [
        ["YOLO", f"{fps_yolo:.1f}", f"{map50_yolo:.3f}", f"{map5095_yolo:.3f}", f"{prec_yolo:.3f}", f"{rec_yolo:.3f}", f"{f1_yolo:.3f}"],
        ["FCOS", f"{fps_fcos:.1f}", f"{map50_fcos:.3f}", f"{map5095_fcos:.3f}", f"{prec_fcos:.3f}", f"{rec_fcos:.3f}", f"{f1_fcos:.3f}"],
    ]

    for r in rows:
        for i, txt in enumerate(r):
            pdf.cell(col_w[i], 8, str(txt), border=1, align="C")
        pdf.ln()

    pdf.ln(4)
    pdf.set_font(pdf.font_family, "", 10)
    pdf.multi_cell(0, 5, f"Speed: YOLO is {speed_ratio:.2f}x faster (higher FPS).")
    pdf.ln(2)
    pdf.multi_cell(0, 5, "Note: per-class metrics may be undefined for rare classes if there are no GT objects or no predictions.")


    # ---------- Page: video notes + conclusions ----------
    pdf.add_page()
    pdf.chapter_title("Section 4 - Video inference notes and conclusions")

    conclusions_text = (
        "Key conclusions:\n"
        f"- YOLO is faster: {fps_yolo:.1f} FPS vs {fps_fcos:.1f} FPS (about {speed_ratio:.2f}x faster).\n"
        f"- YOLO achieves higher accuracy: mAP@50={map50_yolo:.3f} vs {map50_fcos:.3f}, "
        f"mAP@50-95={map5095_yolo:.3f} vs {map5095_fcos:.3f}.\n"
        "- Warm-start fine-tuning leads to fast convergence: most gains happen in early epochs.\n"
        "- Video inference remains challenging because of motion, small objects, and occlusions.\n"
        "- To improve results: add more diverse training frames, tune confidence/IoU thresholds, "
        "and increase the number of samples for rare classes.\n"
    )
    pdf.chapter_body(conclusions_text)

















    pdf_output_path = os.path.join(out_pdf, "Minecraft_world_prediction.pdf")
    pdf.output(pdf_output_path)
    print(f"pdf file saved in : {pdf_output_path}")



    





def test():
    print('Дебил')

if __name__ == '__main__':
    test()
