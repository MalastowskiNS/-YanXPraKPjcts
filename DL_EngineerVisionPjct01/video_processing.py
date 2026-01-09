from ultralytics import YOLO
import cv2
from tqdm import tqdm
##import glob, os


def process_video(model_type, model, input_video_path, output_video_path, confidence_threshold=0.25, imgsz=640):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видеофайл {input_video_path}")
        return

    # Получаем метаданные видео для корректной записи выходного файла.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Инициализируем объект для записи видео с кодеком 'mp4v'.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if model_type == 'YOLO':
        with tqdm(total=total_frames, desc="Анализ игрового процесса") as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break # Видео закончилось
                # Запускаем инференс. `imgsz` обеспечивает консистентность с обучением.
                
                results = model.predict(frame, imgsz=imgsz, conf=confidence_threshold, verbose=False)    
           
                # Ручная отрисовка рамок и меток для полного контроля над цветом
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf, class_id = float(box.conf[0]), int(box.cls[0])
                        label = f"{model.names[class_id]} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            # Записываем обработанный кадр в выходной видеофайл.
                out.write(frame)
                pbar.update(1)


    elif  model_type == 'FCOS':
          with tqdm(total=total_frames, desc="Анализ игрового процесса") as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break # Видео закончилось
                # Запускаем инференс. `imgsz` обеспечивает консистентность с обучением.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(inputs=frame_rgb, return_vis=True)
                vis_rgb = results["visualization"][0]          # RGB
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

                out.write(vis_bgr)
                pbar.update(1) 

    # Освобождаем все ресурсы, чтобы избежать утечек памяти и проблем с файлами.
    cap.release()
    out.release()
    print(f"\nАнализ завершен. Результат сохранен в файл: {output_video_path}")


    

if __name__ == '__main__':
    # Укажите путь к вашему `data.yaml`, который вы создали в Части 1.
    model = YOLO('yolov8n.pt') 
    input_video_path = "dataset/minecraft/video.mp4"
    output_video_path = 'artifacts/videos/yolo_inference.mp4'
    process_video(model, input_video_path, output_video_path, confidence_threshold=0.25, imgsz=640)