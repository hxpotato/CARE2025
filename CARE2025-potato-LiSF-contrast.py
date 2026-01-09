import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from torch.utils.data import random_split
import csv
import numpy as np
import random
import argparse
from pathlib import Path

NON_CONTRAST_MODALITIES = ["T1.nii.gz", "T2.nii.gz", "DWI_800.nii.gz"]
CONTRAST_MODALITIES = [f"GED{i}.nii.gz" for i in range(1, 5)]
ALL_MODALITIES = NON_CONTRAST_MODALITIES + [f"GED{i}.nii.gz" for i in range(1, 5)]
STAGE_LABELS = {"S1": 0, "S2": 1, "S3": 2, "S4": 3}
Cirrhosis_LABELS = {"S1": 0, "S2": 0, "S3": 0, "S4": 1}
Fibrosis_LABELS = {"S1": 1, "S2": 0, "S3": 0, "S4": 0}
STAGE_LABELS_THREE = {"S1": 0, "S2": 1, "S3": 1, "S4": 2}
NUM_SLICE = 6



class MRIDataset(Dataset):
    def __init__(self, root_dir, use_contrast=True, transform=None,
                 target_size=(128, 128), num_slices=NUM_SLICE):
        """
        root_dir: 数据根目录
        use_contrast: 是否使用对比增强模态
        transform: 额外数据变换
        target_size: (H,W) 取切片时的resize目标大小
        num_slices: 选取的连续切片数
        """
        self.root_dir = root_dir
        self.use_contrast = use_contrast
        self.transform = transform
        self.target_size = target_size
        self.num_slices = num_slices
        self.samples = []
        self.modalities = CONTRAST_MODALITIES if use_contrast else NON_CONTRAST_MODALITIES

        for vendor in os.listdir(root_dir):
            vendor_path = os.path.join(root_dir, vendor)
            for filename in os.listdir(vendor_path):
                label_str = filename.split('-')[-1]
                if label_str not in STAGE_LABELS_THREE:
                    continue
                label = STAGE_LABELS_THREE[label_str]
                label_cirrhosis = Cirrhosis_LABELS[label_str]
                label_fibrosis = Fibrosis_LABELS[label_str]
                sample_path = os.path.join(vendor_path, filename)
                self.samples.append((sample_path, label, label_cirrhosis, label_fibrosis))

    def __len__(self):
        return len(self.samples)

    def resize_slice(self, slice_2d):
        import cv2
        slice_resized = cv2.resize(slice_2d.astype(np.float32), self.target_size, interpolation=cv2.INTER_LINEAR)
        return slice_resized

    def __getitem__(self, idx):
        path, label, label_cirrhosis, label_fibrosis = self.samples[idx]

        imgs = []
        masks = []

        for modal in self.modalities:
            img_path = os.path.join(path, modal)
            if os.path.exists(img_path):
                img = nib.load(img_path).get_fdata()
                if img.ndim == 4 and img.shape[-1] == 1:
                    img = np.squeeze(img, axis=-1)
                if img.shape[-1] < img.shape[0]:
                    img = np.transpose(img, (1, 2, 0))

                z_len = img.shape[2]
                mid_idx = z_len // 2
                start_idx = max(0, mid_idx - self.num_slices // 2)
                end_idx = min(z_len, start_idx + self.num_slices)

                # 取连续切片，形状 (H, W, num_slices)
                slices = img[:, :, start_idx:end_idx]

                # 转置为 (num_slices, H, W)
                slices = np.transpose(slices, (2, 0, 1))

                # resize每张切片，归一化
                slices_resized = []
                for slc in slices:
                    slc_resized = self.resize_slice(slc)
                    slc_norm = (slc_resized - slc_resized.mean()) / (slc_resized.std() + 1e-8)
                    slices_resized.append(slc_norm)
                slices_resized = np.stack(slices_resized, axis=0)  # (num_slices, H, W)

                # 如果切片数量不足num_slices，用0填充
                if slices_resized.shape[0] < self.num_slices:
                    pad_num = self.num_slices - slices_resized.shape[0]
                    pad_shape = (pad_num, self.target_size[0], self.target_size[1])
                    pad_array = np.zeros(pad_shape, dtype=np.float32)
                    slices_resized = np.concatenate([slices_resized, pad_array], axis=0)

                mask = 1
            else:
                slices_resized = np.zeros((self.num_slices, *self.target_size), dtype=np.float32)
                mask = -1

            imgs.append(slices_resized)
            masks.append(mask)

        # imgs shape: (modalities, num_slices, H, W)
        volume = np.stack(imgs, axis=0)

        if self.transform:
            volume = self.transform(volume)

        masks = np.array(masks, dtype=np.float32)

        return (
            torch.tensor(volume, dtype=torch.float32),
            torch.tensor(label),
            torch.tensor(masks, dtype=torch.float32),
            torch.tensor(label_cirrhosis),
            torch.tensor(label_fibrosis)
        )


class EVAMRIDataset(Dataset):
    def __init__(self, root_dir, use_contrast=True, transform=None,
                 target_size=(128, 128), num_slices=NUM_SLICE):
        """
        root_dir: 数据根目录
        use_contrast: 是否使用对比增强模态
        transform: 额外数据变换
        target_size: (H,W) 取切片时的resize目标大小
        num_slices: 选取的连续切片数
        """
        self.root_dir = root_dir
        self.use_contrast = use_contrast
        self.transform = transform
        self.target_size = target_size
        self.num_slices = num_slices
        self.samples = []

        # 模态定义，这里示范，替换为你实际变量
        self.modalities = ALL_MODALITIES if CONTRAST_MODALITIES else NON_CONTRAST_MODALITIES

        for vendor in os.listdir(root_dir):
            vendor_path = os.path.join(root_dir, vendor)
            for filename in os.listdir(vendor_path):
                sample_path = os.path.join(vendor_path, filename)

                self.samples.append((sample_path, filename))

    def __len__(self):
        return len(self.samples)

    def resize_slice(self, slice_2d):
        import cv2
        slice_resized = cv2.resize(slice_2d.astype(np.float32), self.target_size, interpolation=cv2.INTER_LINEAR)
        return slice_resized

    def __getitem__(self, idx):
        path, filename = self.samples[idx]

        imgs = []
        masks = []

        for modal in self.modalities:
            img_path = os.path.join(path, modal)
            if os.path.exists(img_path):
                img = nib.load(img_path).get_fdata()
                if img.ndim == 4 and img.shape[-1] == 1:
                    img = np.squeeze(img, axis=-1)
                # 确保(H,W,D)
                if img.shape[-1] < img.shape[0]:
                    img = np.transpose(img, (1, 2, 0))

                z_len = img.shape[2]
                mid_idx = z_len // 2
                start_idx = max(0, mid_idx - self.num_slices // 2)
                end_idx = min(z_len, start_idx + self.num_slices)

                # 取连续切片，形状 (H, W, num_slices)
                slices = img[:, :, start_idx:end_idx]

                # 转置为 (num_slices, H, W)
                slices = np.transpose(slices, (2, 0, 1))

                # resize每张切片，归一化
                slices_resized = []
                for slc in slices:
                    slc_resized = self.resize_slice(slc)
                    slc_norm = (slc_resized - slc_resized.mean()) / (slc_resized.std() + 1e-8)
                    slices_resized.append(slc_norm)
                slices_resized = np.stack(slices_resized, axis=0)  # (num_slices, H, W)

                # 如果切片数量不足num_slices，用0填充
                if slices_resized.shape[0] < self.num_slices:
                    pad_num = self.num_slices - slices_resized.shape[0]
                    pad_shape = (pad_num, self.target_size[0], self.target_size[1])
                    pad_array = np.zeros(pad_shape, dtype=np.float32)
                    slices_resized = np.concatenate([slices_resized, pad_array], axis=0)

                mask = 1
            else:
                slices_resized = np.zeros((self.num_slices, *self.target_size), dtype=np.float32)
                mask = -1

            imgs.append(slices_resized)
            masks.append(mask)

        # imgs shape: (modalities, num_slices, H, W)
        volume = np.stack(imgs, axis=0)

        if self.transform:
            volume = self.transform(volume)

        masks = np.array(masks, dtype=np.float32)

        return (
            torch.tensor(volume, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.float32),
            filename

        )


# --- 2D PatchEmbed ---
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Patch Embedding (Conv-based) ---
# class PatchEmbed3D(nn.Module):
#     def __init__(self, in_channels=1, embed_dim=96, patch_size=4):
#         super().__init__()
#         self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = nn.LayerNorm(embed_dim)
#
#     def forward(self, x):
#         x = self.proj(x)  # [B, C, D, H, W] -> [B, embed_dim, D', H', W']
#         B, C, D, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim], N = D'*H'*W'
#         x = self.norm(x)
#         return x, (D, H, W)



# 更稳健的 PatchEmbed3Dv2
class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels, embed_dim=96):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=2, stride=2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.stem(x)  # [B, C, D', H', W']
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return self.norm(x), (D, H, W)



# --- Swin Transformer Block ---
class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + x_res

        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + x_res
        return x


# --- Deeper SwinTiny3D Backbone ---
class SwinTiny3DDeep(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96, depths=6, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim)
        self.blocks = nn.Sequential(*[
            SwinBlock3D(embed_dim, num_heads=num_heads) for _ in range(depths)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x, _ = self.patch_embed(x)  # [B, N, C]
        x = self.blocks(x)
        x = x.transpose(1, 2)  # [B, C, N]
        x = self.pool(x).squeeze(-1)  # [B, C]
        return x


# --- SE-based Attention Fusion for 3 features ---
class SEFusion(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, f1, f2, f3):
        # f1, f2, f3: [B, C]
        feat = torch.stack([f1, f2, f3], dim=1)  # [B, 3, C]
        mean_feat = feat.mean(dim=1)  # [B, C]
        attn = self.fc(mean_feat).unsqueeze(1)  # [B, 1, C]
        fused = (feat * attn).sum(dim=1)  # [B, C]
        return fused


# --- Residual MLP Head with LayerNorm and GELU ---
class ResidualMLPHead_cirrhosis(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.mlp(x)


# --- Residual MLP Head with LayerNorm and GELU ---
class ResidualMLPHead_fibrosis(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.mlp(x)




# --- Final Dual Head Model with SE Fusion and Residual Heads ---
class SwinDualHeadModel(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96, num_heads=4):
        super().__init__()
        self.encoder_ged1 = SwinTiny3DDeep(in_channels, embed_dim, depths=6, num_heads=num_heads)
        self.encoder_ged2 = SwinTiny3DDeep(in_channels, embed_dim, depths=6, num_heads=num_heads)
        self.encoder_ged3 = SwinTiny3DDeep(in_channels, embed_dim, depths=6, num_heads=num_heads)
        self.encoder_ged4 = SwinTiny3DDeep(in_channels, embed_dim, depths=6, num_heads=num_heads)

        self.se_fusion = SEFusion(embed_dim)

        self.head_cirrhosis = ResidualMLPHead_cirrhosis(embed_dim* 4)
        self.head_fibrosis = ResidualMLPHead_fibrosis(embed_dim* 4)

    def forward(self, x):
        # x: [B, 3, D, H, W], 三个通道分别是 T1, T2, DWI
        ged1 = x[:, 0:1, :, :, :]
        ged2 = x[:, 1:2, :, :, :]
        ged3 = x[:, 2:3, :, :, :]
        ged4 = x[:, 3:4, :, :, :]

        f1 = self.encoder_ged1(ged1)  # [B, embed_dim]
        f2 = self.encoder_ged2(ged2)
        f3 = self.encoder_ged3(ged3)
        f4 = self.encoder_ged4(ged4)

        # feat1 = self.se_fusion(f1, f2, f3)  # [B, embed_dim]
        feat2 = torch.cat([f1, f2, f3,f4], dim=1)  # [B, embed_dim * 3]

        out_cirrhosis = self.head_cirrhosis(feat2)  # [B, 1]
        out_fibrosis = self.head_fibrosis(feat2)  # [B, 1]

        return out_cirrhosis.squeeze(1), out_fibrosis.squeeze(1)


def train_epoch(model, loader, criterion, optimizer, device, task):
    model.train()
    losses, all_preds, all_labels = [], [], []
    loop = tqdm(loader, desc="Training", leave=True)

    for data in loop:
        x, _, _, label_cirrhosis, label_fibrosis = data
        x = x.to(device)
        label_cirrhosis = label_cirrhosis.to(device).float()
        label_fibrosis = label_fibrosis.to(device).float()

        optimizer.zero_grad()
        logits1, logits2 = model(x)

        if task == 'cirrhosis':
            loss = criterion(logits1.squeeze(), label_cirrhosis)
            probs = torch.sigmoid(logits1).detach().cpu().numpy()
            labels = label_cirrhosis.cpu().numpy()
        else:
            loss = criterion(logits2.squeeze(), label_fibrosis)
            probs = torch.sigmoid(logits2).detach().cpu().numpy()
            labels = label_fibrosis.cpu().numpy()

        preds = (probs > 0.5).astype(int)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        all_preds.extend(preds)
        all_labels.extend(labels)

        loop.set_postfix(loss=loss.item())

    acc = accuracy_score(all_labels, all_preds)
    return np.mean(losses), acc


def evaluate(model, loader, device, criterion, task):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            x, _, _, label_cirrhosis, label_fibrosis = data
            x = x.to(device)
            label_cirrhosis = label_cirrhosis.to(device).float()
            label_fibrosis = label_fibrosis.to(device).float()

            logits1, logits2 = model(x)

            if task == 'cirrhosis':
                logits = logits1.squeeze()
                labels = label_cirrhosis
            else:
                logits = logits2.squeeze()
                labels = label_fibrosis

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    print(all_probs)
    print('\n')
    print(all_labels)


    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    print(f"[{task.upper()}] ACC: {acc:.4f} | AUC: {auc:.4f}")
    return acc, auc


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



def main_train(train_dir,timestamp, use_contrast=False, epochs=50, batch_size=25, val_split=0.2, device_id=0, task='cirrhosis'):
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"runs/run_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    dataset_full = MRIDataset(train_dir, use_contrast)

    val_size = int(len(dataset_full) * val_split)
    train_size = len(dataset_full) - val_size
    train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SwinDualHeadModel().to(device)

    if task == 'cirrhosis':
        for p in model.head_fibrosis.parameters():
            p.requires_grad = False
    elif task == 'fibrosis':
        for p in model.head_cirrhosis.parameters():
            p.requires_grad = False
    else:
        raise ValueError(f"Unknown task: {task}")

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    # criterion = nn.BCEWithLogitsLoss()
    if task=='cirrhosis':
        criterion = FocalLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-4)

    best_auc, best_acc = 0, 0
    history = []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, task)
        val_acc, val_auc = evaluate(model, val_loader, device, criterion, task)

        print(
            f"Epoch {epoch + 1}: Loss={train_loss:.4f} | Train ACC={train_acc:.4f} | Val ACC={val_acc:.4f} | AUC={val_auc:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_auc': val_auc
        })

        if val_auc > best_auc and val_auc >0.6 and val_acc>0.55 :
            best_auc = val_auc
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_{task}_epoch_{epoch + 1}.pth'))
            print(f"New best model for {task} saved.")

    pd.DataFrame(history).to_csv(os.path.join(save_dir, f"log_{task}.csv"), index=False)
    print(f"\nBest {task.upper()} Model: AUC={best_auc:.4f}, ACC={best_acc:.4f}")

def evaluate_one(model, loader, device, task='cirrhosis'):
    model.eval()
    results = []

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Evaluating {task}", leave=False):
            x, mask, case_ids = data
            x = x.to(device)

            logits1, logits2 = model(x)
            probs_cirrhosis = torch.sigmoid(logits1).cpu().numpy()
            probs_fibrosis = torch.sigmoid(logits2).cpu().numpy()

            for i in range(len(case_ids)):
                case_id = case_ids[i]
                setting = 'Contrast'

                if task == 'cirrhosis':
                    prob = float(probs_cirrhosis[i])
                    results.append([case_id, setting, round(prob, 5)])
                elif task == 'fibrosis':
                    prob = float(probs_fibrosis[i])
                    results.append([case_id, setting, round(prob, 5)])
                else:
                    raise ValueError(f"Unknown task: {task}")

    if task == 'cirrhosis':
        return pd.DataFrame(results, columns=["Case", "Setting", "Subtask1_prob_S4"])
    else:
        return pd.DataFrame(results, columns=["Case", "Setting", "Subtask2_prob_S1"])



def evaluate_dual_model_and_merge(
        ckpt_cirrhosis_path,
        ckpt_fibrosis_path,
        val_dir,
        batch_size=1,
        device_id=0,
        final_csv_path=None
):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    if final_csv_path is None:
        final_csv_path = Path("./LiFS_pred.csv")
    else:
        final_csv_path = Path(final_csv_path)

        if final_csv_path.suffix != '.csv':
            final_csv_path = final_csv_path / 'LiFS_pred.csv'

    final_csv_path.parent.mkdir(parents=True, exist_ok=True)    # 加载验证集
    val_dataset = EVAMRIDataset(val_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ======== 模型 1: cirrhosis ========
    model1 = SwinDualHeadModel().to(device)
    state_dict1 = torch.load(ckpt_cirrhosis_path, map_location=device, weights_only=True)
    model1.load_state_dict(state_dict1)
    df_cirrhosis = evaluate_one(model1, val_loader, device, task='cirrhosis')

    # ======== 模型 2: fibrosis ========
    model2 = SwinDualHeadModel().to(device)
    state_dict2 = torch.load(ckpt_fibrosis_path, map_location=device, weights_only=True)
    model2.load_state_dict(state_dict2)
    df_fibrosis = evaluate_one(model2, val_loader, device, task='fibrosis')

    # ======== 合并并保存 ========
    df_merged = pd.merge(df_cirrhosis, df_fibrosis, on=["Case", "Setting"], how="inner")
    df_merged.to_csv(final_csv_path, index=False)

    print(f"[✅ Done] Final merged CSV saved to: {final_csv_path}")


# def parse_args():
#     """解析命令行参数"""
#     parser = argparse.ArgumentParser(description="LiQA Medical Image Analysis Pipeline")
#
#     # 必需参数
#     parser.add_argument("--input_dir", type=str, default="/input",
#                         help="Path to input data directory")
#     parser.add_argument("--output_dir", type=str, default="/output",
#                         help="Path to save output results")
#     parser.add_argument("--device_id", type=int, default=0,
#                         help="GPU device ID (use -1 for CPU)")
#
#     return parser.parse_args()

# def main(args):
#     # 创建输出目录
#
#
#     # 设备设置
#
#     # 时间戳（用于结果版本管理）
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     print(f"\n{'=' * 40}")
#     print(f"Input Directory: {args.input_dir}")
#     print(f"Output Directory: {args.output_dir}")
#     print(f"Device: {args.device_id}")
#     print(f"{'=' * 40}\n")
#     # 评估模式
#     evaluate_dual_model_and_merge(
#         ckpt_cirrhosis_path = 'best_model_cirrhosis.pth',
#         ckpt_fibrosis_path = 'best_model_fibrosis.pth',
#         val_dir=args.input_dir,
#         batch_size=1,
#         device_id=args.device_id,
#         final_csv_path = args.output_dir
#     )


if __name__ == "__main__":

    # args = parse_args()

    # # 验证输入路径是否存在
    # if not os.path.exists(args.input_dir):
    #     raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    #
    # main(args)

    # main("/media/xinhong/data/2023SY/LiSF/data/LiQA_training_data", val_dir=None, test_dir=None,use_contrast=False, device_id=1, early_stop=False)  # 关闭早停

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # main_train(r"/data/sde/LiQ/train",timestamp =timestamp,use_contrast=True, device_id=0,task = 'cirrhosis')  # 关闭早停
    # main_train(r"/data/sde/LiQ/train",timestamp =timestamp,use_contrast=True, device_id=0,task = 'fibrosis')  # 关闭早停
    # #

    # evaluate_dual_model_and_merge(
    #     ckpt_cirrhosis_path='/data/sde/LiQ/code/runs/run_20260106_221008/best_model_cirrhosis_epoch_16.pth',
    #     ckpt_fibrosis_path='/data/sde/LiQ/code/runs/run_20260106_221008/best_model_fibrosis_epoch_3.pth',
    #     val_dir='/data/sde/LiQ/LiQA_val/Data',
    #     batch_size=1,
    #     device_id=0,
    #     final_csv_path="/data/sde/LiQ/code")



    evaluate_dual_model_and_merge(
        ckpt_cirrhosis_path='/data/sde/LiQ/code/runs/run_20260108_192643/best_model_cirrhosis_epoch_25.pth',
        ckpt_fibrosis_path='/data/sde/LiQ/code/runs/run_20260108_192643/best_model_fibrosis_epoch_4.pth',
        val_dir='/data/sde/LiQ/LiQA_val/Data',
        batch_size=1,
        device_id=0,
        final_csv_path="/data/sde/LiQ/code")
