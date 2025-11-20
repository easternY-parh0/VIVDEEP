import os
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = "./data/test/raw"           # 입력 폴더
SAVE_DIR = "./data/test/resample"     # 출력 폴더
TARGET_SAMPLES = 8000                 # 목표 timestep

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)


def resample_to_m_points(t, values, M):
    """
    t: (N,) 원본 시간
    values: (N,d) 또는 (N,) 원본 데이터
    M: 목표 sample 수
    """
    t = np.array(t)
    values = np.array(values)

    # endpoint=False → 마지막 t 제외
    t_new = np.linspace(t[0], t[-1], M, endpoint=False)

    # 보간
    if values.ndim == 1:
        v_new = np.interp(t_new, t, values)
    else:
        v_new = np.vstack([
            np.interp(t_new, t, values[:, i]) for i in range(values.shape[1])
        ]).T

    return t_new, v_new


def process_all():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    print(f"[INFO] Found {len(files)} files to resample.\n")

    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        print(f"[INFO] Processing {fname}")

        df = pd.read_csv(path)

        # ===== 시간 컬럼 자동 탐지 =====
        time_col = None
        for c in df.columns:
            if c.lower() in ["t", "time"]:
                time_col = c
                break

        if time_col is None:
            print(f"[WARNING] {fname} skipped (no t column)")
            continue

        # 원본 데이터
        t = df[time_col].values
        data_cols = [c for c in df.columns if c != time_col]
        values = df[data_cols].values

        # ===== 리샘플링 =====
        t_new, values_new = resample_to_m_points(t, values, TARGET_SAMPLES)

        # ===== DataFrame 구성 =====
        out_df = pd.DataFrame(values_new, columns=data_cols)

        # ===== t 값을 0~7999로 강제 재정의 =====
        out_df.insert(0, time_col, np.arange(TARGET_SAMPLES))

        # 저장
        save_path = os.path.join(SAVE_DIR, fname)
        out_df.to_csv(save_path, index=False)

        print(f" → Saved: {save_path}\n")

    print("\n[INFO] All files processed successfully.")


if __name__ == "__main__":
    process_all()
