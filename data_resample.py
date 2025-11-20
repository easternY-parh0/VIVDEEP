import os
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# 설정
# -----------------------------
RAW_DIR = "./data/raw"          # 원본 데이터 폴더
SAVE_DIR = "./data/resample"    # 저장 폴더
TARGET_SAMPLES = 8000           # 재샘플링할 목표 timestep

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------
# 메인 처리
# -----------------------------
def process_all():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

    print(f"[INFO] Found {len(files)} raw data files.\n")

    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        print(f"[INFO] Processing: {fname}")

        df = pd.read_csv(path)

        # 시간 컬럼 이름 자동 감지
        time_col = None
        for c in df.columns:
            if c.lower() in ["t", "time"]:
                time_col = c
                break

        if time_col is None:
            print(f"[WARNING] {fname} skipped (no time column)")
            continue

        original_samples = len(df)
        step = max(original_samples // TARGET_SAMPLES, 1)

        # 지정된 간격으로 샘플 추출
        df_downsampled = df.iloc[::step].copy()

        # 필요 시 마지막 샘플 잘라서 정확히 TARGET_SAMPLES 맞추기
        if len(df_downsampled) > TARGET_SAMPLES:
            df_downsampled = df_downsampled.iloc[:TARGET_SAMPLES]

        # t 컬럼 재설정: 0 ~ TARGET_SAMPLES-1
        df_downsampled[time_col] = np.arange(len(df_downsampled))

        # 저장
        save_name = fname  # 이름 그대로 저장
        df_downsampled.to_csv(os.path.join(SAVE_DIR, save_name), index=False)

        print(f" → Saved: {save_name} | Samples: {len(df_downsampled)}\n")

    print("\n[INFO] All files processed successfully.")


if __name__ == "__main__":
    process_all()
