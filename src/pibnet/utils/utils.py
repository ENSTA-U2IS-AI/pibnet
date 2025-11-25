import numpy as np

def get_obstacle_mask(points: np.ndarray, obstacles: dict, env_dim: int) -> np.ndarray:
    """
    Get the mask of points inside the obstacles.

    Args:
        points (np.ndarray): Array of points to check.
        sample (dict): A dictionary containing obstacle information.

    Returns:
        np.ndarray: Boolean mask indicating which points are inside the obstacles.
    """
    mask = np.zeros(points.shape[0], dtype=bool)
    for obs in obstacles:
        # if obs['shape'] != 'sphere':
        #     raise NotImplementedError(f"Shape {obs['shape']} not implemented")
        center = np.array(obs['center']).reshape(1, -1)[:, :env_dim]
        radius = obs['radius']
        distances = np.linalg.norm(points - center, axis=1)
        mask |= distances <= radius
    return mask


def get_obstacle_mask(points: np.ndarray, obstacles: dict, env_dim: int) -> np.ndarray:
    """
    Get the mask of points inside the obstacles.

    Args:
        points (np.ndarray): Array of points to check.
        sample (dict): A dictionary containing obstacle information.

    Returns:
        np.ndarray: Boolean mask indicating which points are inside the obstacles.
    """
    mask = np.zeros(points.shape[0], dtype=bool)
    for obs in obstacles:
        center = np.array(obs['center']).reshape(1, -1)[:, :env_dim]
        if obs['shape'] == 'sphere':
        #     raise NotImplementedError(f"Shape {obs['shape']} not implemented")
            radius = obs['radius']
            distances = np.linalg.norm(points - center, axis=1)
            mask |= distances <= radius
        elif obs['shape'] == 'ellipsoid':
            radii = np.array(obs['radii']).reshape(1, -1)
            mask |= np.sum(((center - points) ** 2) / (radii ** 2), axis=-1) <= 1

    return mask


def complex_to_real_arrays(complex_arr):
    return np.stack([complex_arr.real, complex_arr.imag], axis=-1)