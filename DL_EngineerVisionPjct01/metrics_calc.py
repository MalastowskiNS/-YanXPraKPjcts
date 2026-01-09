
import os, json, torch


## расчет метрик

def get_gt_info (classes, path_to_json = "dataset/minecraft/annotations/_test_annotations.coco.json"):

    coco = json.load(open(path_to_json, "r", encoding="utf-8"))
    id2name  = {c["id"]: c["name"] for c in coco["categories"]}
    name2label  = {name: i for i, name in enumerate(classes)}
    catid2label = {cid: name2label[id2name[cid]] for cid in id2name}
    gts_by_name = {os.path.basename(im["file_name"]): [] for im in coco["images"]}
    imgid2name  = {im["id"]: os.path.basename(im["file_name"]) for im in coco["images"]}

    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]                  
        label = catid2label[a["category_id"]]   
        gts_by_name[imgid2name[a["image_id"]]].append([x, y, x+w, y+h, label])

    gts_by_name = {k: torch.tensor(v, dtype=torch.float32) for k, v in gts_by_name.items()}

    return gts_by_name


def boxs2tensor(ds, score_thr=0.25):
    # Tensor [N, 6] -> [x1,y1,x2,y2,score,label]
    inst = ds.pred_instances
    bboxes = inst.bboxes
    scores = inst.scores
    labels = inst.labels

    m = scores >= score_thr
    bboxes, scores, labels = bboxes[m], scores[m], labels[m]

    return torch.cat([
        bboxes,
        scores.unsqueeze(1),
        labels.to(torch.float32).unsqueeze(1)
    ], dim=1)

def get_predict_info(interf2rez, score_thr=0.25):
    # получаем datasamples
    preds_by_name = {}

    for ds in interf2rez["predictions"]:
        name = os.path.basename(ds.metainfo["img_path"])
        preds_by_name[name] = boxs2tensor(ds, score_thr=score_thr)
    return preds_by_name


## old one ##
def calculate_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # Распаковка координат и вычисление площадей
    x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1_2, y1_2, x2_2, y2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Нахождение координат пересечения с помощью broadcasting
    inter_x1 = torch.max(x1_1.unsqueeze(1), x1_2.unsqueeze(0))
    inter_y1 = torch.max(y1_1.unsqueeze(1), y1_2.unsqueeze(0))
    inter_x2 = torch.min(x2_1.unsqueeze(1), x2_2.unsqueeze(0))
    inter_y2 = torch.min(y2_1.unsqueeze(1), y2_2.unsqueeze(0))

    # Вычисление площади пересечения и объединения
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

    # Вычисление IoU с защитой от деления на ноль
    iou = inter_area / (union_area + 1e-6)
    return iou


def calculate_ap_for_class(
    pred_boxes: torch.Tensor, 
    true_boxes: torch.Tensor, 
    iou_threshold: float = 0.5
) -> float:
    # Если нет предсказаний или истинных рамок, AP = 0
    if pred_boxes.nelement() == 0:
        return 0.0
    if true_boxes.nelement() == 0:
        return 0.0

    num_true_boxes = true_boxes.shape[0]

    # Сортируем предсказания по убыванию уверенности
    confidences = pred_boxes[:, 4]
    sorted_indices = torch.argsort(confidences, descending=True)
    pred_boxes = pred_boxes[sorted_indices]

    tp = torch.zeros(pred_boxes.shape[0])
    fp = torch.zeros(pred_boxes.shape[0])
    true_boxes_covered = torch.zeros(num_true_boxes, dtype=torch.bool)

    iou_matrix = calculate_iou_matrix(pred_boxes[:, :4], true_boxes)

    for i in range(pred_boxes.shape[0]):
        best_iou_per_pred, best_gt_idx = torch.max(iou_matrix[i], dim=0)

        if best_iou_per_pred > iou_threshold:
            if not true_boxes_covered[best_gt_idx]:
                tp[i] = 1 # True Positive
                true_boxes_covered[best_gt_idx] = True
            else:
                fp[i] = 1 # False Positive (дубликат)
        else:
            fp[i] = 1 # False Positive (низкий IoU)

    # Вычисляем накопительные суммы
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    # Вычисляем Precision и Recall
    recalls = tp_cumsum / (num_true_boxes + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # Добавляем начальную точку (0,0)
    recalls = torch.cat((torch.tensor([0.0]), recalls))
    precisions = torch.cat((torch.tensor([0.0]), precisions))

    # Делаем кривую Precision монотонно убывающей (интерполяция)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])

    # Находим индексы, где Recall меняется
    i_list = torch.where(recalls[1:] != recalls[:-1])[0]

    # Суммируем площади прямоугольников: (R_i - R_{i-1}) * P_i
    ap = torch.sum((recalls[i_list + 1] - recalls[i_list]) * precisions[i_list + 1])

    return ap.item()

def calculate_map(
    predictions: list, 
    ground_truths: list, 
    num_classes: int, 
    iou_threshold: float = 0.5
) -> float:
    average_precisions = []

    for c in range(num_classes):
        all_preds_c = []
        all_gts_c = []

        for i in range(len(predictions)):
            # Предсказания для класса с в изображении i
            preds_in_image = predictions[i]
            class_preds_mask = preds_in_image[:, 5] == c
            for p in preds_in_image[class_preds_mask]:
                all_preds_c.append([i, *p[:5].tolist()]) # Добавляем индекс изображения

            # GT для класса с в изображении i
            gts_in_image = ground_truths[i]
            class_gts_mask = gts_in_image[:, 4] == c
            for g in gts_in_image[class_gts_mask]:
                all_gts_c.append([i, *g[:4].tolist()]) # Добавляем индекс изображения

        num_gts = len(all_gts_c)
        if num_gts == 0:
            continue

        # Если предсказаний для этого класса нет, его AP = 0
        if not all_preds_c:
            average_precisions.append(0.0)
            continue

        # Конвертируем в тензоры
        preds_tensor = torch.tensor(all_preds_c)
        gts_tensor = torch.tensor(all_gts_c)

        # Сортируем предсказания по уверенности
        preds_tensor = preds_tensor[preds_tensor[:, 5].argsort(descending=True)]
        
        # Создаем флаги для отслеживания найденных GT
        # [img_idx, gt_idx_in_image] -> bool
        gt_detected = {} # Используем словарь для удобства

        tp = torch.zeros(preds_tensor.shape[0])
        fp = torch.zeros(preds_tensor.shape[0])

        for i, pred in enumerate(preds_tensor):
            pred_img_idx = pred[0].int().item()

            # Находим все GT для того же изображения, где находится текущий pred
            gts_in_same_image_mask = gts_tensor[:, 0] == pred_img_idx
            gts_in_same_image = gts_tensor[gts_in_same_image_mask]

            # Если в этом изображении нет GT для данного класса, это FP
            if gts_in_same_image.nelement() == 0:
                fp[i] = 1
                continue

            # Рассчитываем IoU только с GT из того же изображения
            ious = calculate_iou_matrix(pred[1:5].unsqueeze(0), gts_in_same_image[:, 1:5])
            best_iou, best_gt_local_idx = torch.max(ious, dim=1)

            # Создаем уникальный ключ для GT: (индекс картинки, локальный индекс GT)
            gt_key = (pred_img_idx, best_gt_local_idx.item())

            if best_iou >= iou_threshold:
                if gt_key not in gt_detected:
                    tp[i] = 1
                    gt_detected[gt_key] = True # Помечаем этот GT как найденный
                else:
                    fp[i] = 1 # Дубликат для уже найденного GT
            else:
                fp[i] = 1 # Низкий IoU

        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        recalls = tp_cumsum / (num_gts + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        recalls = torch.cat((torch.tensor([0.0]), recalls))
        precisions = torch.cat((torch.tensor([0.0]), precisions))

        for j in range(len(precisions) - 2, -1, -1):
            precisions[j] = torch.max(precisions[j], precisions[j+1])
  
        i_list = torch.where(recalls[1:] != recalls[:-1])[0]
        ap = torch.sum((recalls[i_list + 1] - recalls[i_list]) * precisions[i_list + 1])
        average_precisions.append(ap.item())

    if not average_precisions:
        return 0.0

    mean_ap = sum(average_precisions) / len(average_precisions)
    return mean_ap


def calculate_prf1_for_class(
    pred_boxes: torch.Tensor,   # [N,5] = x1 y1 x2 y2 score  (только один класс!)
    true_boxes: torch.Tensor,   # [M,4] = x1 y1 x2 y2        (только один класс!)
    score_thr: float = 0.25,
    iou_threshold: float = 0.5
):
    # сортируем по score
    sorted_idx = torch.argsort(pred_boxes[:, 4], descending=True)
    pred_boxes = pred_boxes[sorted_idx]

    tp = torch.zeros(pred_boxes.shape[0])
    fp = torch.zeros(pred_boxes.shape[0])
    covered = torch.zeros(true_boxes.shape[0], dtype=torch.bool)

    iou_mat = calculate_iou_matrix(pred_boxes[:, :4], true_boxes)  # [N,M]

    for i in range(pred_boxes.shape[0]):
        best_iou, best_gt = torch.max(iou_mat[i], dim=0)
        if best_iou >= iou_threshold and (not covered[best_gt]):
            tp[i] = 1
            covered[best_gt] = True
        else:
            fp[i] = 1

    tp_c = torch.cumsum(tp, 0)
    fp_c = torch.cumsum(fp, 0)

    # берем точку при score_thr: сколько предов прошло порог
    k = int((pred_boxes[:, 4] >= score_thr).sum().item())

    if k == 0:
        TP = 0
        FP = 0
    else:
        TP = int(tp_c[k-1].item())
        FP = int(fp_c[k-1].item())

    FN = int(true_boxes.shape[0] - TP)

    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "tp": TP, "fp": FP, "fn": FN}

def calculate_prf1_all_classes(
    predictions: list,          # list of [Ni,6] = x1 y1 x2 y2 score label
    ground_truths: list,        # list of [Mi,5] = x1 y1 x2 y2 label
    num_classes: int,
    score_thr: float = 0.25,
    iou_threshold: float = 0.5
):
    tp = torch.zeros(num_classes, dtype=torch.float32)
    fp = torch.zeros(num_classes, dtype=torch.float32)
    fn = torch.zeros(num_classes, dtype=torch.float32)

    for i in range(len(predictions)):
        preds = predictions[i]
        gts = ground_truths[i]


        device = gts.device  # <-- главное: выбираем устройство
        preds = preds.to(device)   # <-- переносим GT туда же


        # на всякий случай приведём label к long (бывает float)
        pred_labels = preds[:, 5].long()
        gt_labels = gts[:, 4].long()

        for c in range(num_classes):
            # preds of class c with score >= thr
            pm = (pred_labels == c) & (preds[:, 4] >= score_thr)
            pc = preds[pm]
            if pc.numel() > 0:
                pc = pc[torch.argsort(pc[:, 4], descending=True)]  # sort by score

            # gts of class c
            gm = (gt_labels == c)
            gc = gts[gm]

            # если GT нет — все преды этого класса = FP
            if gc.numel() == 0:
                fp[c] += pc.shape[0]
                continue

            # если предов нет — все GT этого класса = FN
            if pc.numel() == 0:
                fn[c] += gc.shape[0]
                continue

            gt_boxes = gc[:, :4]
            pred_boxes = pc[:, :4]

            Np = pred_boxes.shape[0]
            Ng = gt_boxes.shape[0]

            iou_mat = calculate_iou_matrix(pred_boxes, gt_boxes)  # [Np, Ng]

            # пары (pred, gt), где IoU >= threshold
            pi, gi = torch.where(iou_mat >= iou_threshold)

            if pi.numel() == 0:
                # нет совпадений вообще
                fp[c] += Np
                fn[c] += Ng
            else:
                # сортируем пары по IoU убыванию (как в YOLO)
                ious = iou_mat[pi, gi]
                order = torch.argsort(ious, descending=True)
                pi = pi[order]
                gi = gi[order]

                used_p = torch.zeros(Np, dtype=torch.bool, device=pred_boxes.device)
                used_g = torch.zeros(Ng, dtype=torch.bool, device=gt_boxes.device)

                tpc = 0
                for p_idx, g_idx in zip(pi.tolist(), gi.tolist()):
                    if (not used_p[p_idx]) and (not used_g[g_idx]):
                        used_p[p_idx] = True
                        used_g[g_idx] = True
                        tpc += 1

                tp[c] += tpc
                fp[c] += (Np - used_p.sum().item())
                fn[c] += (Ng - used_g.sum().item())

 
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    # micro (по всем классам)
    TP = tp.sum().item()
    FP = fp.sum().item()
    FN = fn.sum().item()
    p_micro = TP / (TP + FP + 1e-6)
    r_micro = TP / (TP + FN + 1e-6)
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-6)

    return {
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "tp_per_class": tp,
        "fp_per_class": fp,
        "fn_per_class": fn,
        "micro": {"precision": p_micro, "recall": r_micro, "f1": f1_micro, "tp": TP, "fp": FP, "fn": FN}
    }


def calculate_prf1_all_classes_true(
    predictions: list,          # list of tensors [Ni,6] = x1 y1 x2 y2 score label
    ground_truths: list,        # list of tensors [Mi,5] = x1 y1 x2 y2 label
    num_classes: int,
    score_thr: float = 0.25,
    iou_threshold: float = 0.5,
):
    device = ground_truths[0].device if len(ground_truths) > 0 else torch.device("cpu")

    tp = torch.zeros(num_classes, dtype=torch.float32, device=device)
    fp = torch.zeros(num_classes, dtype=torch.float32, device=device)
    fn = torch.zeros(num_classes, dtype=torch.float32, device=device)

    for preds, gts in zip(predictions, ground_truths):
        preds = preds.to(device)
        gts = gts.to(device)

        pred_labels = preds[:, 5].long() if preds.numel() else torch.empty(0, dtype=torch.long, device=device)
        gt_labels   = gts[:, 4].long()   if gts.numel() else torch.empty(0, dtype=torch.long, device=device)

        for c in range(num_classes):
            # --- фильтруем по классу ---
            p = preds[(pred_labels == c) & (preds[:, 4] >= score_thr)]
            g = gts[gt_labels == c]

            num_g = g.shape[0]
            num_p = p.shape[0]

            if num_g == 0 and num_p == 0:
                continue
            if num_g == 0 and num_p > 0:
                fp[c] += num_p
                continue
            if num_g > 0 and num_p == 0:
                fn[c] += num_g
                continue

            # --- сортируем предсказания по score ---
            order = torch.argsort(p[:, 4], descending=True)
            p = p[order]

            # --- матчинг: каждый GT может быть взят максимум 1 раз ---
            matched_gt = torch.zeros(num_g, dtype=torch.bool, device=device)
            iou = calculate_iou_matrix(p[:, :4], g[:, :4])  # [num_p, num_g]

            for i in range(num_p):
                best_iou, best_j = torch.max(iou[i], dim=0)
                if best_iou >= iou_threshold and (not matched_gt[best_j]):
                    tp[c] += 1
                    matched_gt[best_j] = True
                else:
                    fp[c] += 1

            # всё, что не заматчилось — FN
            fn[c] += (~matched_gt).sum().float()

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return precision, recall, f1, tp, fp, fn




def yolo_preds_by_name(model_yolo, img_paths, conf=0.25, iou=0.5, imgsz=640):
    preds_by_name = {}
    for r in model_yolo.predict(
        source=img_paths,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        stream=True,
        verbose=False
    ):
        name = os.path.basename(r.path)
        b = r.boxes
        if b is None or len(b) == 0:
            preds_by_name[name] = torch.zeros((0, 6), dtype=torch.float32)
            continue
        preds_by_name[name] = torch.cat(
            [b.xyxy.float(), b.conf.float().unsqueeze(1), b.cls.float().unsqueeze(1)],
            dim=1
        ).cpu()
    return preds_by_name










    

if __name__ == '__main__':
    path_to_json= "dataset/minecraft/annotations/_test_annotations.coco.json"
    classes = [
        'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 'ghast',
        'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 'turtle', 'wolf', 'zombie'
    ]
    get_gt_info(classes, path_to_json) 