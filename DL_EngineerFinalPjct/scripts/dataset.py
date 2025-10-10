import torch
from torch.utils.data import Dataset

from PIL import Image
import timm
import numpy as np
import pandas as pd

from transformers import AutoTokenizer


import albumentations as A



from matplotlib import pyplot as plt


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, ds_type="train", mask = None):
        if ds_type == "train":
            self.df = pd.read_csv(config.TRAIN_DF_PATH)
        else:
            self.df = pd.read_csv(config.VAL_DF_PATH)

        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME, clean_up_tokenization_spaces=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "ingredients2text"]                                                          # тут подправил
        label = self.df.loc[idx, "total_calories"]                                                           # тут подправил
        mass = self.df.loc[idx, "total_mass"]

        img_path = self.df.loc[idx, "dish_id"]                                                               # тут подправил
                                                                 
        try:
            image = Image.open(f"data/images/{img_path}/rgb.png").convert('RGB')                             # тут подправил
        except:
            image = torch.randint(0, 255, (*self.image_cfg.input_size[1:],
                                           self.image_cfg.input_size[0])).to(
                                               torch.float32)

        image = self.transforms(image=np.array(image))["image"]
        return {"label": label, "image": image, "text": text, "mass": mass}


def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]                                                    
    images = torch.stack([item["image"] for item in batch])
    masses = torch.FloatTensor([item["mass"] for item in batch])                                              
    labels = torch.FloatTensor([item["label"] for item in batch])                                     

    tokenized_input = tokenizer(texts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True)

    return {
        "label": labels,
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        "mass": masses
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.RandomResizedCrop(
                    size=(cfg.input_size[1], cfg.input_size[2]),
                    scale=(0.85, 1.0),
                    ratio=(0.85, 1.0),
                    p=0.8
                                    ),
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.8, 1.2),
                         rotate=(-15, 15),
                         translate_percent=(-0.1, 0.1),
                         shear=(-10, 10),
                         fill=0,
                         p=0.8),
                A.CoarseDropout(
                                num_holes_range=(1, 4),
                                hole_height_range=(int(0.05 * cfg.input_size[1]), int(0.1 * cfg.input_size[1])),
                                hole_width_range=(int(0.05 * cfg.input_size[2]), int(0.1 * cfg.input_size[2])),
                                fill=0,
                                p=0.3
                                ),
                A.ColorJitter(brightness=0.15,
                              contrast=0.15,
                              saturation=0.15,
                              hue=0.015,
                              p=0.6),
                A.Resize(cfg.input_size[1], cfg.input_size[2]),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )

    return transforms
