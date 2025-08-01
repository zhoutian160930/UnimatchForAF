import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score
from models import ecgTransForm
from config import Configs
from augment import WeakAugment1D, StrongAugment1D,NPYDataset
from utils import AverageMeter, setup_logger

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--labeled-ids', type=str, required=True)
    parser.add_argument('--unlabeled-ids', type=str, required=True)
    parser.add_argument('--val-ids', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save-path', type=str, default='./checkpoints')
    parser.add_argument('--pretrained', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    logger = setup_logger(log_path=os.path.join(args.save_path, 'train.log'))
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    weak_aug = WeakAugment1D()
    strong_aug = StrongAugment1D()
    data_path = os.path.join(args.data_root, 'cutdata.npy')

    dataset_l = NPYDataset(args.labeled_ids, data_path, mode='labeled', transform=weak_aug)
    dataset_u_w = NPYDataset(args.unlabeled_ids, data_path, mode='unlabeled', transform=weak_aug)
    dataset_u_s = NPYDataset(args.unlabeled_ids, data_path, mode='unlabeled', transform=strong_aug)
    dataset_val = NPYDataset(args.val_ids, data_path, mode='val', transform=None)

    dataloader_l = DataLoader(dataset_l, batch_size=args.batch_size, shuffle=True)
    dataloader_u_w = DataLoader(dataset_u_w, batch_size=args.batch_size, shuffle=True)
    dataloader_u_s = DataLoader(dataset_u_s, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Loaded datasets: labeled={len(dataset_l)}, unlabeled={len(dataset_u_w)}, val={len(dataset_val)}")

    # === Model ===
    hparams = {
        "feature_dim": Configs.final_out_channels,
    }
    model = ecgTransForm(Configs(), hparams).to(device)
    model_ema = deepcopy(model)
    model_ema.eval()
    for p in model_ema.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    conf_thresh = 0.5
    best_acc_ema = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_u = AverageMeter()

        loader = zip(dataloader_l, zip(dataloader_u_w, dataloader_u_s))

        for i, ((x_l, y_l), (x_u_w, x_u_s)) in enumerate(loader):
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u_w, x_u_s = x_u_w
            x_u_w, x_u_s = x_u_w.to(device), x_u_s.to(device)

            logits_x = model(x_l)
            loss_x = F.cross_entropy(logits_x, y_l)

            with torch.no_grad():
                logits_u_w = model_ema(x_u_w)
                probs_u_w = F.softmax(logits_u_w, dim=1)
                max_probs, pseudo_labels = probs_u_w.max(dim=1)

            logits_u_s = model(x_u_s)
            loss_u_all = criterion(logits_u_s, pseudo_labels)
            mask = (max_probs >= conf_thresh).float()
            loss_u = (loss_u_all * mask).sum() / max(mask.sum(), torch.tensor(1.0, device=device))

            loss = (loss_x + loss_u) / 2
            if i == 0 and epoch == 0:
                logger.info(f"[DEBUG] x_u_w shape: {x_u_w.shape}, x_u_s shape: {x_u_s.shape}")
                logger.info(f"[DEBUG] max_probs: {max_probs.max().item():.4f}, mean: {max_probs.mean().item():.4f}")
                logger.info(f"[DEBUG] mask.sum(): {mask.sum().item()}, mask.mean(): {mask.mean().item():.4f}")
                logger.info(f"[DEBUG] loss_u: {loss_u.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                ema_decay = min(1 - 1 / (i + 1 + epoch * len(dataloader_l)), 0.996)
                for p, p_ema in zip(model.parameters(), model_ema.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p.data * (1 - ema_decay))

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_u.update(loss_u.item())

            if i % 20 == 0:
                logger.info(f"[Epoch {epoch} Iter {i}] Loss={total_loss.avg:.4f} Sup={total_loss_x.avg:.4f} Unsup={total_loss_u.avg:.4f}")

        acc, f1 = evaluate(model, dataloader_val, device)
        acc_ema, f1_ema = evaluate(model_ema, dataloader_val, device)

        logger.info(f"[Epoch {epoch}] ACC={acc:.2f}, F1={f1:.2f} | EMA ACC={acc_ema:.2f}, EMA F1={f1_ema:.2f}")

        checkpoint = {
            'model': model.state_dict(),
            'model_ema': model_ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if acc_ema > best_acc_ema:
            best_acc_ema = acc_ema
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            logger.info(f"Best EMA model saved at epoch {epoch}, acc: {acc_ema:.2f}")

if __name__ == '__main__':
    main()
