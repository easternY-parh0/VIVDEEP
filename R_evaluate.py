import os
import glob
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from R_dataset import VIVDataset
from R_model import MLP
import pandas as pd

def evaluate(checkpoint, data_dir="./data/test/resample", save_dir="./eval_results", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # 체크포인트 로드
    model = MLP(in_dim=3, hidden=256, nlayers=4, out_dim=4).to(device)
    ck = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ck.get("model", ck))
    model.eval()

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if len(csv_files) == 0:
        raise RuntimeError(f"No CSV files in {data_dir}")

    for csv_file in csv_files:
        # CSV별 데이터셋
        dataset = VIVDataset(csv_file)
        t = torch.tensor(dataset.t, dtype=torch.float32).to(device)
        D = torch.tensor(dataset.D, dtype=torch.float32).to(device)
        MR = torch.tensor(dataset.MR, dtype=torch.float32).to(device)
        Y_true = torch.tensor(dataset.Y, dtype=torch.float32).to(device)

        with torch.no_grad():
            inp = torch.stack([t, D, MR], dim=1)
            Y_pred = model(inp).cpu().numpy()
            Y_true_np = Y_true.cpu().numpy()
            t_np = t.cpu().numpy()

        # 그래프 그리기
        outputs = ["d_x","d_y","h_x","h_y"]
        plt.figure(figsize=(12,8))
        for i, name in enumerate(outputs):
            plt.subplot(2,2,i+1)
            plt.plot(t_np, Y_true_np[:,i], label="True")
            plt.plot(t_np, Y_pred[:,i], label="Pred", alpha=0.7)
            plt.xlabel("t (normalized)")
            plt.ylabel(name)
            plt.title(f"{name} vs t")
            plt.grid(True)
            plt.legend()
        plt.tight_layout()

        # 파일명에 CSV 이름 포함
        idx = os.path.splitext(os.path.basename(csv_file))[0]
        save_path = os.path.join(save_dir, f"{idx}_eval_{checkpoint[-9:-4]}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved evaluation plot for {idx} to {save_path}")

        # MSE 계산
        mse = ((Y_pred - Y_true_np)**2).mean(axis=0)
        print(f"MSE for {idx}: {mse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./R_checkpoints/pinn_ckpt_e1000.pth")
    parser.add_argument("--data_dir", type=str, default="./data/test/resample")
    parser.add_argument("--save_dir", type=str, default="./R_eval_results")
    args = parser.parse_args()

    evaluate(checkpoint=args.checkpoint, data_dir=args.data_dir, save_dir=args.save_dir)
