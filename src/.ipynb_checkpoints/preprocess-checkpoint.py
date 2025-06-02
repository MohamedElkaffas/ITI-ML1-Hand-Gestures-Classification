import numpy as np
import pandas as pd

def process_hand_landmarks_xy(row: pd.Series) -> np.ndarray:
    """
    Input: pandas Series of length 63 (x1,y1,z1, …, x21,y21,z21).
    Steps:
      1. Reshape → (21×3), keep only x,y → (21×2)
      2. Re-center by subtracting wrist (row 0)
      3. Normalize by distance to mid‐finger tip (row 11)
      4. Return a flat (42,) vector
    """
    arr = np.array(row).reshape(21, 3)[:, :2].astype(float)  # (21,2)
    wrist = arr[0, :].copy()
    arr = arr - wrist
    mid_tip = arr[11, :].copy()
    scale = np.linalg.norm(mid_tip)
    if scale == 0:
        scale = 1.0
    arr = arr / scale
    return arr.flatten()
