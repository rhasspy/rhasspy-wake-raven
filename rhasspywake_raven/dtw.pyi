from typing import Optional, Tuple, List
import numpy as np

class DynamicTimeWarping:
    cost_matrix: Optional[np.ndarray] = None
    distance: Optional[float] = None
    distance_func: str = "cosine"

    def __init__(self, distance_func: str = "cosine"): ...
    def compute_cost(self, x: np.ndarray, y: np.ndarray, window: Optional[int] = None, **cost_args) -> np.ndarray: ...
    def compute_path(self) -> Optional[List[Tuple[int, int]]]: ...

