import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score

from models import ecgTransForm
from augment import WeakAugment1D, StrongAugment1D
from loguru import logger


class NPYDataset(Dataset):
    def __init__(self, id_file, data_path, mode='labeled', transform=None):
        """
        Args:
            id_file: 包含每行一个 'index label' 的 txt 文件
            data_path: 统一的 .npy 文件路径，例如 'alldata.npy'
            mode: 'labeled', 'unlabeled' 或 'val'
            transform: 可选的变换函数
        """
        self.data = np.load(data_path, mmap_mode='r')  # 高效按需加载
        self.mode = mode
        self.transform = transform

        self.ids = []
        self.labels = []

        with open(id_file, 'r') as f:
            for line in f:
                idx, label = line.strip().split()
                self.ids.append(int(idx))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)  # (1, L)

        if self.transform is not None:
            data = self.transform(data)

        data = data.copy()  
        data = torch.tensor(data, dtype=torch.float32)

        return data, label


def train_one_epoch(model, model_ema, dataloader_l, dataloader_u_pair, 
                    optimizer, criterion, device, epoch, cfg):
    model.train()
    model_ema.eval()

    total_loss, total_sup, total_unsup = 0.0, 0.0, 0.0

    dataloader_u_w, dataloader_u_s = dataloader_u_pair

    for (x_l, y_l), (x_uw, _), (x_us, _) in zip(dataloader_l, dataloader_u_w, dataloader_u_s):
        x_l, y_l = x_l.to(device), y_l.to(device)
        x_uw = x_uw.to(device)
        x_us = x_us.to(device)

        # pseudo label
        with torch.no_grad():
            logits_uw = model_ema(x_uw)
            probs_uw = F.softmax(logits_uw, dim=1)
            max_probs, pseudo_labels = probs_uw.max(dim=1)
            mask = max_probs.ge(cfg['conf_thresh']).float()

        logits_l = model(x_l)
        loss_sup = criterion(logits_l, y_l)

        logits_us = model(x_us)
        loss_unsup = F.cross_entropy(logits_us, pseudo_labels, reduction='none')
        loss_unsup = (loss_unsup * mask).mean()

        loss = loss_sup + cfg['unsup_weight'] * loss_unsup

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update
        for param, ema_param in zip(model.parameters(), model_ema.parameters()):
            ema_param.data.mul_(cfg['ema_decay']).add_(param.data, alpha=1 - cfg['ema_decay'])

        total_loss += loss.item()
        total_sup += loss_sup.item()
        total_unsup += loss_unsup.item()

    return total_loss, total_sup, total_unsup



def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1


def load_ids(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str,required=True)
    parser.add_argument('--labeled-ids', type=str, required=True)
    parser.add_argument('--unlabeled-ids', type=str, required=True)
    parser.add_argument('--val-ids', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save-path', type=str, default='./checkpoints')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights')

    args = parser.parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


    weak_aug = WeakAugment1D()
    strong_aug = StrongAugment1D()
    data_path = os.path.join(args.data_root, 'cutdata.npy')

    dataset_l = NPYDataset(id_file=args.labeled_ids, data_path=data_path, mode='labeled', transform=weak_aug)
    dataset_u_w = NPYDataset(id_file=args.unlabeled_ids, data_path=data_path, mode='unlabeled', transform=weak_aug)
    dataset_u_s = NPYDataset(id_file=args.unlabeled_ids, data_path=data_path, mode='unlabeled', transform=strong_aug)
    dataset_val = NPYDataset(id_file=args.val_ids, data_path=data_path, mode='val', transform=None)
    logger.info(f"Loaded datasets: {len(dataset_l)} labeled, {len(dataset_u_w)} unlabeled (weak), {len(dataset_u_s)} unlabeled (strong), {len(dataset_val)} validation")
    


    dataloader_l = DataLoader(dataset_l, batch_size=args.batch_size, shuffle=True)
    dataloader_u_w = DataLoader(dataset_u_w, batch_size=args.batch_size, shuffle=True)
    dataloader_u_s = DataLoader(dataset_u_s, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
    logger.info(f"Created dataloaders: {len(dataloader_l)} labeled, {len(dataloader_u_w)} unlabeled (weak), {len(dataloader_u_s)} unlabeled (strong), {len(dataloader_val)} validation")

    # 模型初始化
    from config import Configs
    hparams = {
        "feature_dim": Configs.final_out_channels,  # 对应 AdaptiveAvgPool1d(1) 输出后展平的维度
    }
    model = ecgTransForm(Configs(), hparams).to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    model.load_state_dict(torch.load(args.pretrained)['model'])
    model_ema = deepcopy(model)
    for p in model_ema.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    cfg = {
        'conf_thresh': 0.9,
        'unsup_weight': 0.95,
        'ema_decay': 0.95
    }

    best_f1 = 0

    for epoch in range(args.epochs):
        loss, loss_sup, loss_unsup = train_one_epoch(   model, model_ema,
                                                        dataloader_l,
                                                        (dataloader_u_w, dataloader_u_s),
                                                        optimizer, criterion,
                                                        device, epoch, cfg
                                                    )
        acc, f1 = evaluate(model_ema, dataloader_val, device)

        print(f"[Epoch {epoch}] Loss: {loss:.4f}, Sup: {loss_sup:.4f}, Unsup: {loss_unsup:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(model_ema.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
            print(f"Best model saved @ epoch {epoch} with F1: {f1:.4f}")


if __name__ == '__main__':
    main()
