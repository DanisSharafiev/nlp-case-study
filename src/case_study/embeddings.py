from pathlib import Path

import numpy as np


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32, copy=False)
