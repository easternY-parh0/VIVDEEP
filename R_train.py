import os, argparse, gc, time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from R_dataset import VIVDataset
from R_model import MLP
from R_physics import compute_physics_residual

def collate_fn(batch):
    # stack items to tensors efficiently
    t = torch.tensor(np.array([b["t"] for b in batch]), dtype=torch.float32)
    D = torch.tensor(np.array([b["D"] for b in batch]), dtype=torch.float32)
    MR = torch.tensor(np.array([b["MR"] for b in batch]), dtype=torch.float32)
    y = torch.tensor(np.array([b["y"] for b in batch]), dtype=torch.float32)
    M = torch.tensor(np.array([b["M"] for b in batch]), dtype=torch.float32)
    K = torch.tensor(np.array([b["K"] for b in batch]), dtype=torch.float32)
    C = torch.tensor(np.array([b["C"] for b in batch]), dtype=torch.float32)
    return {"t": t, "D": D, "MR": MR, "y": y, "M": M, "K": K, "C": C}

def train_pinn(data_dir: str,
               out_ckpt: str = "pinn_ckpt.pth",
               epochs: int = 1000,
               batch_size: int = 2048,
               lr: float = 1e-3,
               physics_weight: float = 1.0,
               device: str = None,
               resume: str = None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = VIVDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = MLP(in_dim=3, hidden=256, nlayers=4, out_dim=4).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    start_epoch = 0
    if resume and os.path.isfile(resume):
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_epoch = ck.get("epoch", 0) + 1
        print(f"Resumed from {resume}, starting epoch {start_epoch}")

    mse = nn.MSELoss()

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_phys_loss = 0.0
        n = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            t = batch["t"].to(device)
            D = batch["D"].to(device)
            MR = batch["MR"].to(device)
            y = batch["y"].to(device)
            M = batch["M"].to(device)
            K = batch["K"].to(device)
            C = batch["C"].to(device)

            # compute preds and physics residuals
            preds, resid = compute_physics_residual(
                model, t, D, MR, M, K, C,
                device=device
            )

            data_loss = mse(preds, y)
            phys_loss = resid.pow(2).mean()
            loss = data_loss + physics_weight * phys_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_phys_loss += phys_loss.item()
            n += 1

        avg_loss = total_loss / n
        avg_data = total_data_loss / n
        avg_phys = total_phys_loss / n
        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1:04d} | Loss {avg_loss:.6e} | Data {avg_data:.6e} | Phys {avg_phys:.6e} | LR {optimizer.param_groups[0]['lr']:.2e}")
        #print("Sleeping for 30s...")
        #time.sleep(30)

        # === checkpoint 저장 ===
        if (epoch+1) % 10 == 0 or epoch == epochs - 1:

            # out_ckpt = "checkpoints/pinn_ckpt.pth" 라면 "checkpoints/" 생성
            ckpt_dir = os.path.dirname(out_ckpt)
            if ckpt_dir != "" and not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)

            # 파일명: pinn_ckpt_e0050.pth 같은 형식
            base = os.path.splitext(out_ckpt)[0]   # "checkpoints/pinn_ckpt"
            ckpt_path = f"{base}_e{epoch+1:04d}.pth"

            torch.save({
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epoch,
                "t_mean": dataset.t_mean, "t_std": dataset.t_std,
                "D_mean": dataset.D_mean, "D_std": dataset.D_std,
                "MR_mean": dataset.MR_mean, "MR_std": dataset.MR_std
            }, ckpt_path)

            print(f"Saved checkpoint to {ckpt_path}")

    # final cleanup
    del model, optimizer
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/resample")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--physics_weight", type=float, default=1.0)
    parser.add_argument("--out_ckpt", type=str, default="R_checkpoints/pinn_ckpt.pth")
    parser.add_argument("--resume", type=str, default="R_checkpoints/pinn_ckpt_e0880.pth")
    args = parser.parse_args()

    train_pinn(data_dir=args.data_dir,
               out_ckpt=args.out_ckpt,
               epochs=args.epochs,
               batch_size=args.batch_size,
               lr=args.lr,
               physics_weight=args.physics_weight,
               resume=args.resume)
