import pandas as pd
import numpy as np

def process_hand_landmarks_xy(row: pd.Series) -> np.ndarray:
    """
    Process hand landmarks to extract normalized XY coordinates.
    
    Args:
        row: pandas Series containing 63 values (21 landmarks × 3 coordinates)
        
    Returns:
        Flattened array of 42 normalized XY coordinates
    """
    coords = row.values.reshape(21, 3)[:, :2].copy()
    wrist = coords[0]
    coords -= wrist
    mid_tip = coords[11]          # landmark 12 (index 11) = middle-finger tip
    scale = np.linalg.norm(mid_tip) or 1.0
    coords /= scale
    return coords.flatten()