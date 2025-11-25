import numpy as np
from typing import Any, Dict


def generate_source(cfg: Any) -> Dict[str, Any]:
    """
    Generates the source which define the boundary conditions.

    Parameters
    ----------
        cfg: dataset config
    """

    source = {}
    source_type = cfg.source.type
    source['type'] = source_type

    if cfg.equation == 'helmholtz':
        if source_type == 'monopole':
            position = np.random.uniform(0., cfg.environment.size, cfg.environment.dim)
            if cfg.environment.dim == 2:
                position = np.concatenate([position, np.array([0])])
            source['position'] = position.tolist()

        elif source_type == 'plane':
            source_dir = np.random.uniform(-1., 1., cfg.environment.dim)
            source_dir /= np.linalg.norm(source_dir)
            if cfg.environment.dim == 2:
                source_dir = np.concatenate([source_dir, np.array([0])])
            source['direction'] = source_dir.tolist()

        else:
            raise NotImplementedError
        
        source['amplitude'] = 1.
        
    elif cfg.equation == 'laplace':
        position = np.random.uniform(0., cfg.environment.size, cfg.environment.dim)
        if cfg.environment.dim == 2:
            position = np.concatenate([position, np.array([0])])
        source['position'] = position.tolist()

        source_dir = np.random.uniform(-1., 1., cfg.environment.dim)
        source_dir /= np.linalg.norm(source_dir)
        if cfg.environment.dim == 2:
            source_dir = np.concatenate([source_dir, np.array([0])])
        source['direction'] = source_dir.tolist()
        source['amplitudes'] = (np.random.dirichlet([1] * 3) * np.random.choice([-1, 1], size=3)).tolist()

    else:
        raise NotImplementedError

    source['wavelength'] = np.random.uniform(cfg.source.wavelength.min, cfg.source.wavelength.max) * cfg.obstacles.h

    return source


def generate_sphere_obs(cfg: Any) -> Dict[str, Any]:
    """
    Generates the parameters for a spherical obstacle.

    Parameters
    ----------
        cfg: dataset config
    """
    shape = 'sphere'
    radius = np.random.uniform(cfg.obstacles.sphere.radius.min, cfg.obstacles.sphere.radius.max)
    center = np.random.uniform(0., cfg.environment.size, cfg.environment.dim)
    if cfg.environment.dim == 2:
        center = np.concatenate([center, np.array([0])])
    return {
        'shape': shape,
        'radius': radius,
        'center': center.tolist()
    }


def generate_ellipsoid_obs(cfg: Any) -> Dict[str, Any]:
    """
    Generates the parameters for a ellipsoidal obstacle.

    Parameters
    ----------
        cfg: dataset config
    """
    shape = 'ellipsoid'
    radii = np.random.uniform(cfg.obstacles.ellipsoid.radius.min, cfg.obstacles.ellipsoid.radius.max, cfg.environment.dim)
    center = np.random.uniform(0., cfg.environment.size, cfg.environment.dim)
    if cfg.environment.dim == 2:
        center = np.concatenate([center, np.array([0])])
    return {
        'shape': shape,
        'radii': radii.tolist(),
        'center': center.tolist()
    }

def get_max_radius(obstacle: dict) -> float:
    """
    Gets the max radius of an obstacle.

    Parameters
    ----------
        obstacle: the obstacle parameters

    """
    if obstacle['shape'] == 'sphere':
        return obstacle['radius']
    
    elif obstacle['shape'] == 'ellipsoid':
        return max(obstacle['radii'])
    
    else:
        raise NotImplementedError("obstacle shape must be one of ['sphere', 'ellipsoid']")

def check_obstacles_overlap(new_obs: dict, obs_list: list, margin: float = 0.) -> bool:
    """
    Checks if a new obstacle overlaps with any obstacles in a given list.

    Parameters
    ----------
        new_obs: dict
            New obstacle parameters
        obs_list: list
        A list of dictionaries, where each dictionary represents an obstacle with its parameters.
        margin: float, optional 
            An additional margin to consider when checking for overlaps, by defaults 0.
    """
    new_center = np.array(new_obs['center'])
    new_radius = get_max_radius(new_obs)
    for obs in obs_list:
        obs_center = np.array(obs['center'])

        centers_distance = np.linalg.norm(new_center - obs_center)
        if centers_distance < (new_radius + get_max_radius(obs) + margin):
            return True
        
    return False


def check_source_overlap(obstacle: dict, source: dict, margin: float = 0.) -> bool:
    """
    Checks whether a monopole source overlaps with an obstacle.

    Parameters
    ----------
    obstacle : dict
        The obstacle parameters
    source : dict
        The source parameters
    margin : float, optional
        An additional margin to consider when checking for overlap, by defaults 0.
    """
    if source['type'] == 'monopole':
        if obstacle['shape'] == 'sphere':
            center = np.array(obstacle['center'])
            source_pos = np.array(source['position'])
            return np.linalg.norm(center - source_pos) < (obstacle['radius'] + margin)
        
        elif obstacle['shape'] == 'ellipsoid':
            center = np.array(obstacle['center'])
            source_pos = np.array(source['position'])
            return np.sum(((center - source_pos) ** 2) / ((np.array(obstacle['radii']) + margin) ** 2)) < 1
        
        else:
            raise NotImplementedError("obstacle shape must be one of ['sphere', 'ellipsoid']")
    
    else:
        return False


def generate_sample_param(cfg: Any) -> Dict[str, Any]:
    """
    Generates the parameters of a dataset sample.

    Parameters
    ----------
        cfg: dataset config
    """
    source = generate_source(cfg)
    # source['position'] = [5., 5., 0.]
    obstacles = []
    for _ in range(np.random.randint(cfg.obstacles.number.min, cfg.obstacles.number.max)):
        for _ in range(cfg.obstacles.num_tries):
            # obs = generate_sphere_obs(cfg)
            obs = generate_ellipsoid_obs(cfg)
            # obs['center'][-1] = 0.
            if (not check_obstacles_overlap(obs, obstacles, margin=cfg.inter_obs_margin)) and \
               (not check_source_overlap(obs, source, margin=cfg.source_obs_margin)):
                obstacles.append(obs)
                break
    return {
        'source': source,
        'obstacles': obstacles
    }