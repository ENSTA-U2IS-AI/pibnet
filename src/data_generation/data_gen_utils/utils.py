import os
import json
import numpy as np
from typing import Optional
from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def get_obstacle_mask(points: np.ndarray, obstacles: list, env_dim: int) -> np.ndarray:
    """
    Get the mask of points inside the obstacles.

    Parameters
    ----------
        points: np.ndarray 
            Array of points to check.
        obstacles:
            The list of the obstacle parameters.
    """
    mask = np.zeros(points.shape[0], dtype=bool)
    for obs in obstacles:
        if obs['shape'] == 'sphere':
            center = np.array(obs['center']).reshape(1, -1)[:, :env_dim]
            radius = obs['radius']
            distances = np.linalg.norm(points - center, axis=1)
            mask |= distances <= radius
            
        elif obs['shape'] == 'ellipsoid':
            center = np.array(obs['center']).reshape(1, -1)[:, :env_dim]
            radii = obs['radii']
            distances = np.linalg.norm(points - center, axis=1)
            mask |= np.sum(((points - center) ** 2) / (np.array(obs['radii']) ** 2)) <= 1

        else:
            raise NotImplementedError(f"Shape {obs['shape']} not implemented")
        
    return mask