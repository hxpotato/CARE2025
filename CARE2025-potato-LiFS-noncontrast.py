# =====================================================================
#         CARE2025 Challenge – Liver Fibrosis Staging (LiFS)
#         Competition Information & Code Metadata
#         Website: https://zmic.org.cn/care_2025/
#         Team Name: potato
#         Affiliation: HuaQiao University
#         Contributor：Xin Hong,Nao Wang,Ying Shi

#          Version: v1.0.0
#          Last Updated: 2025-08-05
#          Copyright © 2025 potato Team
# This code is intended exclusively for research and competition use.
# =====================================================================

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

NON_CONTRAST_MODALITIES = ["T1.nii.gz", "T2.nii.gz", "DWI_800.nii.gz"]
ALL_MODALITIES = NON_CONTRAST_MODALITIES + [f"GED{i}.nii.gz" for i in range(1, 5)]
STAGE_LABELS = {"S1": 0, "S2": 1, "S3": 2, "S4": 3}
Cirrhosis_LABELS = {"S1": 0, "S2": 0, "S3": 0, "S4": 1}
Fibrosis_LABELS = {"S1": 1, "S2": 0, "S3": 0, "S4": 0}
STAGE_LABELS_THREE = {"S1": 0, "S2": 1, "S3": 1, "S4": 2}

class MRIDataset(Dataset):
    def __init__(self, root_dir, use_contrast=False, transform=None,
                 target_size=(128,128), num_slices=10):
        """
        root_dir
        use_contrast
        transform
        target_size: (H,W) 
        num_slices
        """
        self.root_dir = root_dir
        self.use_contrast = use_contrast
        self.transform = transform
        self.target_size = target_size
        self.num_slices = num_slices
        self.samples = []

        self.modalities = ALL_MODALITIES if use_contrast else NON_CONTRAST_MODALITIES

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
                # (H,W,D)
                if img.shape[-1] < img.shape[0]:
                    img = np.transpose(img, (1, 2, 0))

                z_len = img.shape[2]
                mid_idx = z_len // 2
                start_idx = max(0, mid_idx - self.num_slices // 2)
                end_idx = min(z_len, start_idx + self.num_slices)
                slices = img[:, :, start_idx:end_idx]

                slices = np.transpose(slices, (2, 0, 1))

                slices_resized = []
                for slc in slices:
                    slc_resized = self.resize_slice(slc)
                    slc_norm = (slc_resized - slc_resized.mean()) / (slc_resized.std() + 1e-8)
                    slices_resized.append(slc_norm)
                slices_resized = np.stack(slices_resized, axis=0)  # (num_slices, H, W)
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
    def __init__(self, root_dir, use_contrast=False, transform=None,
                 target_size=(128,128), num_slices=10):
        """
        root_dir: 
        use_contrast: 
        transform: 
        target_size: (H,W) 
        num_slices: 
        """
        self.root_dir = root_dir
        self.use_contrast = use_contrast
        self.transform = transform
        self.target_size = target_size
        self.num_slices = num_slices
        self.samples = []

        self.modalities = ALL_MODALITIES if use_contrast else NON_CONTRAST_MODALITIES

        for vendor in os.listdir(root_dir):
            vendor_path = os.path.join(root_dir, vendor)
            for filename in os.listdir(vendor_path):
                sample_path = os.path.join(vendor_path, filename)

                self.samples.append((sample_path,filename))

    def __len__(self):
        return len(self.samples)

    def resize_slice(self, slice_2d):
        import cv2
        slice_resized = cv2.resize(slice_2d.astype(np.float32), self.target_size, interpolation=cv2.INTER_LINEAR)
        return slice_resized

    def __getitem__(self, idx):
        path,filename = self.samples[idx]

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

                slices = img[:, :, start_idx:end_idx]
                slices = np.transpose(slices, (2, 0, 1))

                slices_resized = []
                for slc in slices:
                    slc_resized = self.resize_slice(slc)
                    slc_norm = (slc_resized - slc_resized.mean()) / (slc_resized.std() + 1e-8)
                    slices_resized.append(slc_norm)
                slices_resized = np.stack(slices_resized, axis=0)  # (num_slices, H, W)

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
class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, C, D, H, W] -> [B, embed_dim, D', H', W']
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim], N = D'*H'*W'
        x = self.norm(x)
        return x, (D, H, W)

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
class ResidualMLPHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.norm = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 1)
        self.residual = nn.Linear(in_dim, 128) if in_dim != 128 else nn.Identity()

    def forward(self, x):
        out = self.fc1(x)
        out = self.norm(out)
        out = F.gelu(out)
        out = out + self.residual(x)  
        out = self.fc2(out)
        return out

# --- Final Dual Head Model with SE Fusion and Residual Heads ---
class SwinDualHeadModel(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96, num_heads=4):
        super().__init__()
        self.encoder_t1 = SwinTiny3DDeep(in_channels, embed_dim, depths=6, num_heads=num_heads)
        self.encoder_t2 = SwinTiny3DDeep(in_channels, embed_dim, depths=6, num_heads=num_heads)
        self.encoder_dwi = SwinTiny3DDeep(in_channels, embed_dim, depths=6, num_heads=num_heads)

        self.se_fusion = SEFusion(embed_dim)

        self.head_cirrhosis = ResidualMLPHead(embed_dim)
        self.head_fibrosis = ResidualMLPHead(embed_dim)

    def forward(self, x):
        # x: [B, 3, D, H, W],  T1, T2, DWI
        t1 = x[:, 0:1, :, :, :]
        t2 = x[:, 1:2, :, :, :]
        dwi = x[:, 2:3, :, :, :]

        f1 = self.encoder_t1(t1)  # [B, embed_dim]
        f2 = self.encoder_t2(t2)
        f3 = self.encoder_dwi(dwi)

        feat = self.se_fusion(f1, f2, f3)  # [B, embed_dim]

        out_cirrhosis = self.head_cirrhosis(feat)  # [B, 1]
        out_fibrosis = self.head_fibrosis(feat)    # [B, 1]

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

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    print(f"[{task.upper()}] ACC: {acc:.4f} | AUC: {auc:.4f}")
    return acc, auc


def main(train_dir, use_contrast=False, epochs=50, batch_size=4, val_split=0.1, device_id=0, task='cirrhosis'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    # criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-4)

    best_auc, best_acc = 0, 0
    history = []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, task)
        val_acc, val_auc = evaluate(model, val_loader, device, criterion, task)

        print(f"Epoch {epoch+1}: Loss={train_loss:.4f} | Train ACC={train_acc:.4f} | Val ACC={val_acc:.4f} | AUC={val_auc:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_auc': val_auc
        })

        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_{task}.pth'))
            print(f"New best model for {task} saved.")

    pd.DataFrame(history).to_csv(os.path.join(save_dir, f"log_{task}.csv"), index=False)
    print(f"\nBest {task.upper()} Model: AUC={best_auc:.4f}, ACC={best_acc:.4f}")


def evaluate_one(model, loader, device, save_path, task='cirrhosis'):
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
                setting = 'NonContrast'

                if task == 'cirrhosis':
                    prob = float(probs_cirrhosis[i])
                    results.append([case_id, setting, round(prob, 5)])
                elif task == 'fibrosis':
                    prob = float(probs_fibrosis[i])
                    results.append([case_id, setting, round(prob, 5)])
                else:
                    raise ValueError(f"Unknown task: {task}")

    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(save_path, mode="w", newline='') as f:
        writer = csv.writer(f)
        if task == 'cirrhosis':
            writer.writerow(["Case", "Setting", "Subtask1_prob_S4"])
        else:
            writer.writerow(["Case", "Setting", "Subtask2_prob_S1"])
        writer.writerows(results)

    print(f"[✅ Saved] {task.capitalize()} predictions saved to: {save_path}")


def evaluate_dual_model_and_merge(
    ckpt_cirrhosis_path,
    ckpt_fibrosis_path,
    val_dir,
    batch_size=1,
    device_id=0,
    final_csv_path="predictions.csv"
):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    val_dataset = EVAMRIDataset(val_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ========  1: cirrhosis ========
    model1 = SwinDualHeadModel().to(device)
    state_dict1 = torch.load(ckpt_cirrhosis_path, map_location=device, weights_only=True)
    model1.load_state_dict(state_dict1)

    temp1 = "cirrhosis_temp.csv"
    evaluate_one(model1, val_loader, device, save_path=temp1, task='cirrhosis')

    # ========  2: fibrosis ========
    model2 = SwinDualHeadModel().to(device)
    state_dict2 = torch.load(ckpt_fibrosis_path, map_location=device, weights_only=True)
    model2.load_state_dict(state_dict2)

    temp2 = "fibrosis_temp.csv"
    evaluate_one(model2, val_loader, device, save_path=temp2, task='fibrosis')

    # ======== merge CSV ========
    df1 = pd.read_csv(temp1)
    df2 = pd.read_csv(temp2)
    df_merged = pd.merge(df1, df2, on=["Case", "Setting"], how="inner")
    df_merged.to_csv(final_csv_path, index=False)

    print(f"[✅ Done] Final merged CSV saved to: {final_csv_path}")

if __name__ == "__main__":

    main(r"/data/sdd/LiQA/Normalization",use_contrast=False, device_id=0,task = 'cirrhosis')  
    main(r"/data/sdd/LiQA/Normalization",use_contrast=False, device_id=0,task = 'fibrosis')  

    eva(ckpt_path = '/data/sdd/LiQA/code/runs/c/best_model_cirrhosis.pth',val_dir='/data/sdd/LiQA/val_data_Normalization',task = 'cirrhosis')
    eva(ckpt_path = '/data/sdd/LiQA/code/runs/f/best_model_fibrosis.pth',val_dir='/data/sdd/LiQA/val_data_Normalization',task = 'fibrosis')

    
    evaluate_dual_model_and_merge(ckpt_cirrhosis_path = '/data/sdd/LiQA/code/runs/c/best_model_cirrhosis.pth',
    ckpt_fibrosis_path= '/data/sdd/LiQA/code/runs/f/best_model_fibrosis.pth',
    val_dir='/data/sdd/LiQA/val_data_Normalization',
    batch_size=1,
    device_id=0,
    final_csv_path="predictions.csv")


    evaluate_dual_model_and_merge(ckpt_cirrhosis_path = '/data/sdd/LiQA/code/runs/run_20250710_170529/best_model_cirrhosis.pth',
    ckpt_fibrosis_path= '/data/sdd/LiQA/code/runs/run_20250710_172214/best_model_fibrosis.pth',
    val_dir='/data/sdd/LiQA/val_data_Normalization',
    batch_size=1,
    device_id=0,

    final_csv_path="predictions.csv")


