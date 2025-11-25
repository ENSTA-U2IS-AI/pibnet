import os
os.environ["DGLBACKEND"] = "pytorch"
from typing import Tuple, Any
import dgl
import random
import numpy as np
import bempp_cl.api
import meshio
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import json
from datetime import datetime
import warnings
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

from data_gen_utils.mesh_utils import mesh_to_graph
from data_gen_utils.sample_utils import generate_sample_param
from data_gen_utils.bempp_utils import generate_grids_list, grids_to_mesh, save_grid
from data_gen_utils.utils import timeit

from data_gen_utils.groundtruth.helmholtz_dirichlet_monopole import generate_groundtruth_helmholtz_dirichlet_monopole
from data_gen_utils.groundtruth.helmholtz_neumann_plane import generate_groundtruth_helmholtz_neumann_plane, generate_groundtruth_helmholtz_neumann_plane_v2
from data_gen_utils.groundtruth.laplace_dirichlet import generate_groundtruth_laplace_dirichlet


def gen_sample_param(index: int, cfg: DictConfig) -> Tuple[dict, dgl.DGLGraph]:
    """
    Generates the parameters of the dataset sample at a given index.

    Parameters
    ----------
        cfg: dataset config
    """
    random.seed(index + cfg.seed)
    np.random.seed(index + cfg.seed)
    sample = generate_sample_param(cfg)
    return sample

# @timeit
def create_sample(index: int, sample_param: dict, cfg: Any):
    """
    Create a dataset sample.

    Parameters
    ----------
        index: int
            The sample index
        sample: dict
            The sample parameters
        cfg: dataset config
    """

    # Do nothing if sample already exists
    if os.path.isfile(os.path.join(cfg.data_path, 'graph', f"{str(index).zfill(6)}.bin")):
        return
    
    # Generate a list of the obstacle grids
    grids = generate_grids_list(sample_param, cfg, cfg.obstacles.h)
    # Merge the grids
    grid = bempp_cl.api.grid.union(grids)
    
    # Generate the groud truth solution on the boundaries depending on the equation
    if cfg.equation == 'helmholtz' and cfg.source.type == 'monopole':
        gt = generate_groundtruth_helmholtz_dirichlet_monopole(grid, index, sample_param, cfg)
    elif cfg.equation == 'helmholtz' and cfg.source.type == 'plane':
        gt = generate_groundtruth_helmholtz_neumann_plane(grid, index, sample_param, cfg)
    elif cfg.equation == 'laplace':
        gt = generate_groundtruth_laplace_dirichlet(grid, index, sample_param, cfg)
    else:
        raise NotImplementedError

    # Convert the obstacle grids to a mesh
    mesh = grids_to_mesh(grids)
    # Convert mesh and ground truth to DGL graph format
    graph = mesh_to_graph(mesh, gt)
    # Save graph
    dgl.save_graphs(os.path.join(cfg.data_path, 'graph', f"{str(index).zfill(6)}.bin"), graph)
    
    
@timeit
@hydra.main(version_base=None, config_path="../../configs/data_generation", config_name="data_gen")
def gen_dataset(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dir_name = cfg.save_name if cfg.save_name else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_path = os.path.join(cfg.save_path, dir_name)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.join(data_path, 'mesh'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'graph'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'boundary_sol'), exist_ok=True)

    with open_dict(cfg):
        cfg['dir_name'] = dir_name
        cfg['data_path'] = data_path

    if cfg.seed is None:
        with open_dict(cfg):
            cfg['seed'] = np.random.randint(1000000)
            print(f"seed: {cfg['seed']}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.save_path is None:
        warnings.warn("Dataset will not be saved !")
    
    try:
        with open(os.path.join(cfg.data_path, "samples.json")) as f:
            samples = json.load(f)
        print('Samples already generated')

    except:
        num_workers = max(1, cpu_count() - 4)
        worker_func = partial(gen_sample_param, cfg=cfg)
        with Pool(processes=num_workers) as pool:
            samples = list(pool.map(worker_func, range(cfg.num_samples)))
        print(samples)

    if cfg.save_path is not None:
        with open(os.path.join(data_path, 'samples.json'), 'w') as fp:
            json.dump(samples, fp)

    num_workers = 4 #max(1, cpu_count() // 4 - 1)
    worker_func = partial(create_sample, cfg=cfg)
    with Pool(processes=num_workers) as pool:
        pool.starmap(worker_func, zip(range(cfg.num_samples), samples))

    # for index, sample in enumerate(samples):
    #     create_sample(index, sample, cfg)

if __name__ == "__main__":
    gen_dataset()