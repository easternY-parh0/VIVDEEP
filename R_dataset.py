import os, glob, re, math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

FILENAME_PATTERN = re.compile(
    r"D(?P<D>\d+)_MR(?P<MR>[\d\.]+)_U0(?P<U0>[\d\.eE+-]+)_NU(?P<NU>[\d\.eE+-]+)"
)

def parse_params_from_filename(fname: str):
    m = FILENAME_PATTERN.search(os.path.basename(fname))
    if not m:
        raise ValueError(f"Filename invalid: {fname}")
    return float(m.group("D")), float(m.group("MR")), float(m.group("U0")), float(m.group("NU").rstrip('.'))

class VIVDataset(Dataset):
    """
    Loads CSVs in data_dir, flattens into timepoint samples.
    Normalizes t, D, MR (stores means/stds for inverse transforms).
    Provides M,K,C (physical parameters) per sample (not normalized).
    """
    def __init__(self, data_dir: str, UR=5.0, DR=0.0):
        if os.path.isfile(data_dir):
            files = [data_dir]
        else:
            files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if len(files) == 0:
            raise RuntimeError(f"No CSV files in {data_dir}")

        samples = []
        for f in files:
            D, MR, U0, NU = parse_params_from_filename(f)
            df = pd.read_csv(f)

            required = {"t","d_x","d_y","h_x","h_y"}
            if not required.issubset(df.columns):
                raise RuntimeError(f"{f} missing columns {required - set(df.columns)}")

            t = df["t"].astype(float).to_numpy()
            dx = df["d_x"].astype(float).to_numpy()
            dy = df["d_y"].astype(float).to_numpy()
            hx = df["h_x"].astype(float).to_numpy()
            hy = df["h_y"].astype(float).to_numpy()

            M = math.pi * (D / 2.0) ** 2 * MR
            FN = U0 / (UR * D)
            K = (FN * 2 * math.pi) ** 2 * M * (1 + 1.0 / MR)
            C = 2 * math.sqrt(K * M) * DR

            for i in range(len(t)):
                samples.append({
                    "t": float(t[i]),
                    "D": float(D),
                    "MR": float(MR),
                    "d_x": float(dx[i]),
                    "d_y": float(dy[i]),
                    "h_x": float(hx[i]),
                    "h_y": float(hy[i]),
                    "M": float(M),
                    "K": float(K),
                    "C": float(C)
                })

        df_all = pd.DataFrame(samples)
        
        # normalization stats
        self.t_mean = float(df_all["t"].mean())
        self.t_std  = float(df_all["t"].std()) if df_all["t"].std() > 0 else 1.0
        self.D_mean = float(df_all["D"].mean())
        self.D_std  = float(df_all["D"].std()) if df_all["D"].std() > 0 else 1.0
        self.MR_mean= float(df_all["MR"].mean())
        self.MR_std = float(df_all["MR"].std()) if df_all["MR"].std() > 0 else 1.0

        # raw inputs
        self.t = df_all["t"].to_numpy(dtype=np.float32)
        self.D = df_all["D"].to_numpy(dtype=np.float32)
        self.MR= df_all["MR"].to_numpy(dtype=np.float32)

        # targets
        self.Y = df_all[["d_x","d_y","h_x","h_y"]].to_numpy(dtype=np.float32)

        # physics params (not normalized)
        self.M_arr = df_all["M"].to_numpy(dtype=np.float32)
        self.K_arr = df_all["K"].to_numpy(dtype=np.float32)
        self.C_arr = df_all["C"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return {
            "t": self.t[idx],
            "D": self.D[idx],
            "MR": self.MR[idx],
            "y": self.Y[idx],
            "M": self.M_arr[idx],
            "K": self.K_arr[idx],
            "C": self.C_arr[idx]
        }
