import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from R_dataset import VIVDataset
from R_model import MLP

def evaluate(checkpoint: str, data_dir: str = "./data", device=None, save_fig=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = VIVDataset(data_dir)
    
    # 모든 데이터 가져오기
    X_t = torch.tensor(dataset.t, dtype=torch.float32).to(device)
    X_D = torch.tensor(dataset.D, dtype=torch.float32).to(device)
    X_MR = torch.tensor(dataset.MR, dtype=torch.float32).to(device)
    Y_true = torch.tensor(dataset.Y, dtype=torch.float32).to(device)

    model = MLP(in_dim=3, hidden=256, nlayers=4, out_dim=4).to(device)
    
    # 체크포인트 로드
    ck = torch.load(checkpoint, map_location=device)
    state = ck.get("model", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        inp = torch.stack([X_t, X_D, X_MR], dim=1)
        Y_pred = model(inp).cpu().numpy()
        Y_true = Y_true.cpu().numpy()
        t = X_t.cpu().numpy()

    # 그래프 그리기 (4개 출력 각각)
    output_names = ["d_x", "d_y", "h_x", "h_y"]
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(t, Y_true[:, i], label="True", color="blue")
        plt.plot(t, Y_pred[:, i], label="Pred", color="red", alpha=0.7)
        plt.xlabel("t")
        plt.ylabel(output_names[i])
        plt.title(f"{output_names[i]} vs t")
        plt.grid(True)
        plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=200, bbox_inches='tight')
        print(f"Saved figure to {save_fig}")
    else:
        plt.show()

    # 간단한 MSE 계산
    mse = np.mean((Y_pred - Y_true)**2, axis=0)
    print("MSE per output (d_x,d_y,h_x,h_y):", mse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./R_checkpoints/pinn_ckpt_e1000.pth") #, required=True
    parser.add_argument("--data_dir", type=str, default="./data/test/resample")
    parser.add_argument("--save_fig", type=str, default="./eval_results/2")
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        save_fig=args.save_fig
    )
