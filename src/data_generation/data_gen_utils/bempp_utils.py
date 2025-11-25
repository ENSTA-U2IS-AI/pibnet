import os
import bempp_cl.api
import numpy as np
from typing import Any, List
import meshio

from data_gen_utils.mesh_utils import merge_meshes
from data_gen_utils.utils import timeit
import time

# @timeit
def generate_grids_list(sample: dict, cfg: Any, h: float=None) -> List[bempp_cl.api.Grid]:
    """
    Generate a list of Bempp grids based on the obstacles in the given sample.

    Parameters
    ----------
        sample: dict
            The sample parameters
        cfg: dataset config
        h: float
            the grid size
    """
    if h is None:
        h=cfg.obstacles.h

    grids = []
    for obs in sample['obstacles']:
        if obs['shape'] == 'sphere':
            grid = bempp_cl.api.shapes.sphere(r=obs['radius'], origin=obs['center'], h=h)
        elif obs['shape'] == 'ellipsoid':
            grid = bempp_cl.api.shapes.ellipsoid(r1=obs['radii'][0], r2=obs['radii'][1], r3=obs['radii'][2], origin=obs['center'], h=h)
        else:
            raise NotImplementedError
        grids.append(grid)

    return grids


def save_grid(grid, index, cfg):
    mesh_path = os.path.join(cfg.save_path, cfg.dir_name, f"mesh/{str(index).zfill(6)}.msh")
    bempp_cl.api.export(mesh_path, grid=grid)


def bempp_grid_to_mesh(grid: bempp_cl.api.Grid) -> meshio.Mesh:
    """
    Convert a bempp_cl grid into a meshio mesh.

    Parameters
    ----------
        grid:bempp_cl.api.Grid
            The bempp grid to convert
    """
    points = grid.vertices.T # Transpose to match meshio format
    num_points = points.shape[0]
    points_group = np.zeros(num_points)

    edges = grid.edges.T
    num_edges = edges.shape[0]
    edges_group = np.zeros(num_edges)

    triangles = grid.elements.T
    num_triangles = triangles.shape[0]
    triangles_group = np.zeros(num_triangles)

    # Create a meshio mesh
    return meshio.Mesh(
        points=points, 
        cells={
            "line": edges,
            "triangle": triangles
        },
        point_data={"group": points_group},
        cell_data={"group": [edges_group, triangles_group]}
    )


def grids_to_mesh(grids: List[bempp_cl.api.Grid]) -> meshio.Mesh:
    """
    Convert a list of bempp_cl grids into a meshio mesh.

    Parameters
    ----------
        grids: List[bempp_cl.api.Grid]
            The list of bempp grids
    """
    meshes = [bempp_grid_to_mesh(grid) for grid in grids]
    return merge_meshes(meshes)