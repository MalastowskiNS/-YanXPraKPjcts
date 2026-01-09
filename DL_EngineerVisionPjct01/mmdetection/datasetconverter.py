# взять с  https://blog.roboflow.com/how-to-convert-annotations-from-voc-xml-to-coco-json/ 

import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re
import hashlib


def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r', encoding='utf-8') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotation ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r', encoding='utf-8') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid + ext_with_dot) for aid in ann_ids]
    return ann_paths


def _stable_image_id_from_filename(filename: str) -> int:
    """
    Делает стабильный int id из имени файла, без коллизий "вытащили цифры из имени".
    COCO требует, чтобы image.id был int.
    """
    h = hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]
    return int(h, 16)


def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)

    # Нормализуем file_name (только basename, без папок)
    filename = os.path.basename(filename)
    img_name = os.path.basename(filename)
    img_id_str = os.path.splitext(img_name)[0]

    # !!! ФИКС: extract_num_from_imgid оставляем как параметр,
    # но если извлечь число нельзя/опасно — используем стабильный hash id.
    img_id = None
    if extract_num_from_imgid and isinstance(img_id_str, str):
        nums = re.findall(r'\d+', img_id_str)
        if len(nums) == 1:
            # это всё равно может коллидировать между split'ами,
            # но оставляем как совместимость, если у тебя имена типа 000123.jpg
            img_id = int(nums[0])
        else:
            img_id = _stable_image_id_from_filename(filename)
    else:
        img_id = _stable_image_id_from_filename(filename)

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,   # важно: basename, без "test/" и т.п.
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def _clip_bbox_xyxy_to_xywh(xmin, ymin, xmax, ymax, W, H):
    """
    VOC: xmin,ymin,xmax,ymax  -> COCO: x,y,w,h
    + клип по границам и проверка на нулевые/отрицательные.
    """
    # clamp corners
    xmin = max(0, min(xmin, W - 1))
    ymin = max(0, min(ymin, H - 1))
    xmax = max(0, min(xmax, W))
    ymax = max(0, min(ymax, H))

    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return None
    return [float(xmin), float(ymin), float(w), float(h)]


def get_coco_annotation_from_obj(obj, label2id, image_width: int, image_height: int):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]

    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin')))
    ymin = int(float(bndbox.findtext('ymin')))
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))

    # (Оставляем твою логику -1 для xmin/ymin как "совместимость",
    # но это может давать -1 -> потом клипнется в 0)
    xmin = xmin - 1
    ymin = ymin - 1

    # Вместо assert сразу клипуем — так ты не потеряешь датасет из-за пары плохих боксов
    bbox = _clip_bbox_xyxy_to_xywh(xmin, ymin, xmax, ymax, image_width, image_height)
    if bbox is None:
        return None

    o_width, o_height = bbox[2], bbox[3]
    ann = {
        'area': float(o_width * o_height),
        'iscrowd': 0,
        'bbox': bbox,
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }

    bnd_id = 1  # annotation id
    print('Start converting !')

    # Для защиты от дублей image_id:
    used_image_ids = set()
    filename_to_id = {}

    removed_boxes = 0
    clipped_boxes = 0

    for a_path in tqdm(annotation_paths):
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)

        filename = img_info["file_name"]
        W, H = img_info["width"], img_info["height"]

        # гарантируем уникальный id картинки внутри этого json
        img_id = img_info["id"]
        if img_id in used_image_ids:
            # если коллизия — делаем id по filename
            img_id = _stable_image_id_from_filename(filename)
            # если и так совпало (очень маловероятно) — добавим соль
            salt = 1
            while img_id in used_image_ids:
                img_id = _stable_image_id_from_filename(f"{filename}__{salt}")
                salt += 1

        used_image_ids.add(img_id)
        filename_to_id[filename] = img_id
        img_info["id"] = img_id

        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id,
                                               image_width=W, image_height=H)
            if ann is None:
                removed_boxes += 1
                continue

            # если исходный bbox вылезал — мы его клипнули (косвенно)
            # (точно считать clipped можно, если сравнивать до/после; здесь оставим упрощённо)
            # clipped_boxes += ...

            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id += 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w', encoding='utf-8') as f:
        json.dump(output_json_dict, f, ensure_ascii=False)

    print(f"Done. Removed invalid boxes: {removed_boxes}. (Clipping is applied when needed.)")


def main():
    parser = argparse.ArgumentParser(
        description='This script support converting voc format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotation files directory. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_ids', type=str, default=None,
                        help='path to annotation files ids list. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_paths_list', type=str, default=None,
                        help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
    parser.add_argument('--labels', type=str, default=None,
                        help='path to label list.')
    parser.add_argument('--output', type=str, default='output.json', help='path to output json file')
    parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
    args = parser.parse_args()

    label2id = get_label2id(labels_path=args.labels)
    ann_paths = get_annpaths(
        ann_dir_path=args.ann_dir,
        ann_ids_path=args.ann_ids,
        ext=args.ext,
        annpaths_list_path=args.ann_paths_list
    )
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=args.output,
        extract_num_from_imgid=True
    )


if __name__ == '__main__':
    main()
