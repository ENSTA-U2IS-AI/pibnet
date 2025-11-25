import os
from typing import Any, List
import numpy as np
import bempp_cl.api

from data_gen_utils.utils import get_obstacle_mask, timeit

def generate_groundtruth_helmholtz_dirichlet_monopole(grid, index: int, sample: dict, cfg: Any, unit_sphere_points: np.ndarray = None):

    # Define space
    piecewise_const_space = bempp_cl.api.function_space(grid, "P", 1)

    # Define incident field
    s_pos = np.array(sample['source']['position'])
    Ampl = sample['source']['amplitude']
    k = 2 * np.pi / sample['source']['wavelength']
        
    @bempp_cl.api.complex_callable
    def u_inc(x, n, domain_index, result):
        r = np.linalg.norm(x - s_pos)
        result[0] = Ampl * np.exp(1j * k * r) / r
        
    grid_fun = bempp_cl.api.GridFunction(piecewise_const_space, fun=u_inc)

    lhs = slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(
        piecewise_const_space, piecewise_const_space, piecewise_const_space, k
    )

    dirichlet_fun, info, iteration_count = bempp_cl.api.linalg.gmres(lhs, -grid_fun, tol=cfg.gmres_tolerance, return_iteration_count=True)
    coefs = dirichlet_fun.coefficients

    if cfg.generate_all_sol:

        # # Generate the far field solution
        # slp_ff = bempp_cl.api.operators.far_field.helmholtz.single_layer(piecewise_const_space, unit_sphere_points, k)

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
        slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(piecewise_const_space, obs_free_points, k)

        r = np.linalg.norm(obs_free_points - s_pos.reshape(-1, 1), axis=0)

        free_field = Ampl * np.exp(1j * k * r) / r

        res = free_field + slp_pot.evaluate(dirichlet_fun)
        u_evaluated[obs_free_mask] = res.flat

        full_sol_path = os.path.join(cfg.save_path, cfg.dir_name, f"full_sol/{str(index).zfill(6)}")
        np.save(full_sol_path, u_evaluated.reshape((Nx, Ny)).T)

        num_iter_path = os.path.join(cfg.save_path, cfg.dir_name, f"num_iter/{str(index).zfill(6)}")
        with open(num_iter_path, "w") as f:
            f.write(str(iteration_count))

    return np.vstack((coefs.real, coefs.imag)).T