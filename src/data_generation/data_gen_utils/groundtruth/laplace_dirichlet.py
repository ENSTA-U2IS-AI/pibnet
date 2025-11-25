import os
from typing import Any, List
import numpy as np
import bempp_cl.api

from data_gen_utils.utils import get_obstacle_mask, timeit

def generate_groundtruth_laplace_dirichlet(grid, index: int, sample: dict, cfg: Any, unit_sphere_points: np.ndarray = None):

    # Define space
    piecewise_const_space = bempp_cl.api.function_space(grid, "P", 1)

    # Define incident field
    s_pos = np.array(sample['source']['position'])
    s_dir = np.array(sample['source']['direction'])
    phi =  np.array(sample['source']['amplitudes'])
        
    @bempp_cl.api.real_callable
    def u_inc(x, n, domain_index, result):
        r = np.linalg.norm(x - s_pos)
        result[0] = phi[0] - phi[1] / r - 2 * phi[2] * np.dot(s_dir, (x-s_pos) / r)
        # result[0] = - 1 / r
        
    grid_fun = bempp_cl.api.GridFunction(piecewise_const_space, fun=u_inc)

    lhs = slp = bempp_cl.api.operators.boundary.laplace.single_layer(
        piecewise_const_space, piecewise_const_space, piecewise_const_space
    )

    dirichlet_fun, info, iteration_count = bempp_cl.api.linalg.gmres(lhs, -grid_fun, tol=cfg.gmres_tolerance, return_iteration_count=True)
    coefs = dirichlet_fun.coefficients

    if cfg.generate_all_sol:

        # # Generate the far field solution
        # slp_ff = bempp_cl.api.operators.far_field.laplace.single_layer(piecewise_const_space, unit_sphere_points)

        # ff_sol = slp_ff.evaluate(dirichlet_fun)

        # ff_sol_path = os.path.join(cfg.save_path, cfg.dir_name, f"far_field/{str(index).zfill(6)}")
        # np.save(ff_sol_path, ff_sol)

        # Space grid discretization for the visu solution
        Nx = cfg.num_point_full_sol
        Ny = cfg.num_point_full_sol
        xmin, xmax, ymin, ymax = 0, cfg.environment.size, 0, cfg.environment.size
        plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
        points = np.vstack((plot_grid[0].ravel(),
                            plot_grid[1].ravel(),
                            s_pos[-1] * np.ones(plot_grid[0].size)))
        u_evaluated = np.zeros(points.shape[1], dtype=np.complex128)
        u_evaluated[:] = np.nan

        obs_free_mask = ~get_obstacle_mask(points.T, sample['obstacles'], env_dim=3)
        obs_free_points = points[:, obs_free_mask]

        # Generate the potential solution on the grid for visu
        slp_pot = bempp_cl.api.operators.potential.laplace.single_layer(piecewise_const_space, obs_free_points)

        r = np.linalg.norm(obs_free_points - s_pos.reshape(-1, 1), axis=0)
        free_field = - phi[1] / r - 2 * phi[2] * np.sum(s_dir.reshape(-1, 1) * (obs_free_points - s_pos.reshape(-1, 1)) / r, axis=0)

        res = free_field + slp_pot.evaluate(dirichlet_fun)
        u_evaluated[obs_free_mask] = res.flat

        full_sol_path = os.path.join(cfg.save_path, cfg.dir_name, f"full_sol/{str(index).zfill(6)}")
        np.save(full_sol_path, u_evaluated.reshape((Nx, Ny)).T)

        num_iter_path = os.path.join(cfg.save_path, cfg.dir_name, f"num_iter/{str(index).zfill(6)}")
        with open(num_iter_path, "w") as f:
            f.write(str(iteration_count))

    return coefs.T