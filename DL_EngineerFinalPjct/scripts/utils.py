import os
import random
from functools import partial

import numpy as np

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from torchmetrics.regression import MeanAbsoluteError

from transformers import AutoModel, AutoTokenizer

from dataset import MultimodalDataset, collate_fn, get_transforms


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        self.mass_proj = nn.Linear(1, config.HIDDEN_DIM*2)
        nn.init.xavier_uniform_(self.mass_proj.weight, gain=0.1)
        nn.init.zeros_(self.mass_proj.bias)


        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM*2, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.HIDDEN_DIM, 1)                                                                     # тут подправил
        )
           

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:,  0, :]
        image_features = self.image_model(image)
 
        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        fused = torch.cat([text_emb, image_emb], dim=-1)
        mass_gate = torch.sigmoid(self.mass_proj(mass.unsqueeze(-1)))  # [B, 2H]
        fused = fused * (1 + 1.75 * (mass_gate - 0.5))  # немягкая модуляция                                      # по лайту сделали    
        out = self.regressor(fused).squeeze(-1)
        return out                                                                                       


def train(config, device, baseline_mae, mask):                                                                                       # тут подправил     mask
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME, clean_up_tokenization_spaces=True)

    
    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)
    

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable}/{n_total} ({100 * n_trainable / n_total:.2f}%)")

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.text_proj.parameters(),
        'lr': 1e-3
    }, {
        'params': model.image_proj.parameters(),
        'lr': 1e-3
    }, {
        'params': model.mass_proj.parameters(),
        'lr': 1e-3
    }, {
        'params': model.regressor.parameters(),
        'lr': config.REGRESSOR_LR
    }], weight_decay=1e-4)



    criterion = nn.MSELoss()                                                                                  # тут подправил                                                       
    
    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    train_dataset = MultimodalDataset(config, transforms, "train", mask)                                               # тут подправил                                          
    val_dataset = MultimodalDataset(config, val_transforms, "val", mask)                                               # тут подправил     
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))

    # инициализируем метрику
    mae_train = MeanAbsoluteError().to(device)                                                                     # тут подправил
    mae_val   = MeanAbsoluteError().to(device)                                                                     # тут подправил
    best_mae_val = baseline_mae
    print("training started")
    vmae2plot = []
    tmae2plot = []
    tmse2plot = []
    for epoch in range(config.EPOCHS):
        model.train()
        model.text_model.train()
        model.image_model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                "mass": batch["mass"].to(device)
            }
            labels = batch['label'].float().to(device)                                                               # тут подправил      

            # Forward
            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits, labels)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

             # === Разнормализуем прямо по батчу ===
            preds_real = logits.detach().cpu() 
            labels_real = labels.detach().cpu()                                  
            _ = mae_train(preds=preds_real, target=labels_real)

        # Валидация
        mae2train = mae_train.compute().cpu().numpy()
        mae2val = validate(model, val_loader, device, mae_val)
        mae_val.reset()
        mae_train.reset()

        print(
            f"Epoch {epoch}/{config.EPOCHS-1} | avg_mse: {total_loss/len(train_loader):.4f} | Train MAE: {mae2train :.4f}| Val MAE: {mae2val :.4f}"
        )

        tmse2plot.append(total_loss/len(train_loader))
        tmae2plot.append(mae2train)
        vmae2plot.append(mae2val)

        if mae2val < best_mae_val:                                                                                      # тут подправил
            print(f"New best model, epoch: {epoch}")
            best_mae_val = mae2val
            torch.save({ 
                        "model_state_dict": model.state_dict(),
                        },               
                       config.SAVE_PATH)
    return tmae2plot, vmae2plot, tmse2plot


def validate(model, val_loader, device, mae_metric, return_labels=False):
    model.eval()
    model.text_model.eval()
    model.image_model.eval()    
    mae_metric.reset()
    all_predicted= []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                "mass": batch["mass"].to(device)
            }
            labels = batch['label'].float().to(device)                                                                 # тут подправил  

            logits = model(**inputs)
            preds_real = logits.detach().cpu() 
            labels_real = labels.detach().cpu()
            _ = mae_metric(preds=preds_real, target=labels_real)

            if return_labels:
                all_predicted.append(preds_real)



    if return_labels:
        return mae_metric.compute().cpu().numpy(), torch.cat(all_predicted).numpy()
    else:
        return mae_metric.compute().cpu().numpy()

