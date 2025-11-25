import os
import bempp_cl.api
import numpy as np
from typing import Any, List
import meshio

from utils.utils import get_obstacle_mask

def gen_far_field_bempp(bound_sol_pred, unit_sphere_points, index: int, sample: dict, cfg: Any) -> np.ndarray:

    k = 2 * np.pi / sample['source']['wavelength']

    mesh_path = os.path.join(cfg.dataset_root, cfg.dataset_test, f"mesh/{str(index).zfill(6)}.msh")
    grid = bempp_cl.api.import_grid(mesh_path)
    piecewise_const_space = bempp_cl.api.function_space(grid, "P", 1)

    grid_fun = bempp_cl.api.GridFunction(piecewise_const_space, coefficients=bound_sol_pred[..., 0] + 1j * bound_sol_pred[..., 1])

    slp_ff = bempp_cl.api.operators.far_field.helmholtz.single_layer(piecewise_const_space, unit_sphere_points, k)

    return slp_ff.evaluate(grid_fun)[0]

def gen_full_sol_pred_bempp_laplace(bound_sol, index: int, sample: dict, cfg: Any, **kwargs) -> np.ndarray:

    # Define acoustic source
    s_pos = np.array(sample['source']['position'])
    s_dir = np.array(sample['source']['direction'])
    phi = np.array(sample['source']['amplitudes'])

    mesh_path = os.path.join(cfg.dataset_root, cfg.dataset_test, f"mesh/{str(index).zfill(6)}.msh")
    grid = bempp_cl.api.import_grid(mesh_path)
    piecewise_const_space = bempp_cl.api.function_space(grid, "P", 1)

    bound_sol = bound_sol.astype(np.float64).reshape(-1)
    grid_fun = bempp_cl.api.GridFunction(piecewise_const_space, coefficients=bound_sol)

    # Space discretization for the solution
    Nx = cfg.test.num_point_full_sol
    Ny = cfg.test.num_point_full_sol
    xmin, xmax, ymin, ymax = 0, cfg.data.environment.size, 0, cfg.data.environment.size
    plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
    points = np.vstack((plot_grid[0].ravel(),
                        plot_grid[1].ravel(),
                        s_pos[-1] * np.ones(plot_grid[0].size)))
    u_evaluated = np.zeros(points.shape[1])
    u_evaluated[:] = np.nan

    obs_free_mask = ~get_obstacle_mask(points.T, sample['obstacles'], env_dim=3).T
    obs_free_points = points[:, obs_free_mask]

    slp_pot = bempp_cl.api.operators.potential.laplace.single_layer(piecewise_const_space, obs_free_points)

    r = np.linalg.norm(obs_free_points - s_pos.reshape(-1, 1), axis=0)
    free_field = - phi[1] / r - 2 * phi[2] * np.sum(s_dir.reshape(-1, 1) * (obs_free_points - s_pos.reshape(-1, 1)) / r, axis=0)

    res = free_field + slp_pot.evaluate(grid_fun)
    u_evaluated[obs_free_mask] = res.flat

    # full_sol_path = os.path.join(cfg.dataset_root, cfg.dataset_test, f"full_sol/{str(index).zfill(6)}")
    # np.save(full_sol_path, u_evaluated.reshape((Nx, Ny)).T)

    return u_evaluated.reshape((Nx, Ny)).T

def gen_full_sol_pred_bempp_helmholtz_dirichlet(bound_sol, index: int, sample: dict, cfg: Any, **kwargs) -> np.ndarray:

    # Define acoustic source
    sourse_pos = np.array(sample['source']['position'])

    k = 2 * np.pi / sample['source']['wavelength']

    mesh_path = os.path.join(cfg.dataset_root, cfg.dataset_test, f"mesh/{str(index).zfill(6)}.msh")
    grid = bempp_cl.api.import_grid(mesh_path)
    piecewise_const_space = bempp_cl.api.function_space(grid, "P", 1)

    grid_fun = bempp_cl.api.GridFunction(piecewise_const_space, coefficients=bound_sol[..., 0] + 1j * bound_sol[..., 1])

    # Space discretization for the solution
    Nx = cfg.test.num_point_full_sol
    Ny = cfg.test.num_point_full_sol
    xmin, xmax, ymin, ymax = 0, cfg.data.environment.size, 0, cfg.data.environment.size
    plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
    points = np.vstack((plot_grid[0].ravel(),
                        plot_grid[1].ravel(),
                        sourse_pos[-1] * np.ones(plot_grid[0].size)))
    u_evaluated = np.zeros(points.shape[1], dtype=np.complex128)
    u_evaluated[:] = np.nan

    obs_free_mask = ~get_obstacle_mask(points.T, sample['obstacles'], env_dim=3).T
    obs_free_points = points[:, obs_free_mask]

    slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(piecewise_const_space, obs_free_points, k)

    r = np.linalg.norm(obs_free_points - sourse_pos.reshape(-1, 1), axis=0)

    free_field = np.exp(1j * k * r) / r

    res = free_field + slp_pot.evaluate(grid_fun)
    u_evaluated[obs_free_mask] = res.flat

    # full_sol_path = os.path.join(cfg.dataset_root, cfg.dataset_test, f"full_sol/{str(index).zfill(6)}")
    # np.save(full_sol_path, u_evaluated.reshape((Nx, Ny)).T)

    return u_evaluated.reshape((Nx, Ny)).T

def gen_full_sol_pred_bempp_helmholtz_neumann(bound_sol, index: int, sample: dict, cfg: Any, **kwargs) -> np.ndarray:

    # Define acoustic source
    sourse_pos = np.array(sample['source']['position'])

    k = 2 * np.pi / sample['source']['wavelength']

    mesh_path = os.path.join(cfg.dataset_root, cfg.dataset_test, f"mesh/{str(index).zfill(6)}.msh")
    grid = bempp_cl.api.import_grid(mesh_path)
    piecewise_const_space = bempp_cl.api.function_space(grid, "P", 1)

    grid_fun = bempp_cl.api.GridFunction(piecewise_const_space, coefficients=bound_sol[..., 0] + 1j * bound_sol[..., 1])

    # Space discretization for the solution
    Nx = cfg.test.num_point_full_sol
    Ny = cfg.test.num_point_full_sol
    xmin, xmax, ymin, ymax = 0, cfg.data.environment.size, 0, cfg.data.environment.size
    plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
    points = np.vstack((plot_grid[0].ravel(),
                        plot_grid[1].ravel(),
                        sourse_pos[-1] * np.ones(plot_grid[0].size)))
    u_evaluated = np.zeros(points.shape[1], dtype=np.complex128)
    u_evaluated[:] = np.nan

    obs_free_mask = ~get_obstacle_mask(points.T, sample['obstacles'], env_dim=3).T
    obs_free_points = points[:, obs_free_mask]

    slp_pot = bempp_cl.api.operators.potential.helmholtz.double_layer(piecewise_const_space, obs_free_points, k)

    inc_dir = np.array(sample['source']['direction'])
    free_field = np.exp(1j * k * np.sum(inc_dir.reshape(-1, 1) * obs_free_points, axis=0))

    res = free_field + slp_pot.evaluate(grid_fun)
    u_evaluated[obs_free_mask] = res.flat

    # full_sol_path = os.path.join(cfg.dataset_root, cfg.dataset_test, f"full_sol/{str(index).zfill(6)}")
    # np.save(full_sol_path, u_evaluated.reshape((Nx, Ny)).T)

    return u_evaluated.reshape((Nx, Ny)).T
