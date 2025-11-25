import os
from typing import Any, List
import numpy as np
import bempp_cl.api

from data_gen_utils.utils import get_obstacle_mask, timeit

def generate_groundtruth_helmholtz_combined_plane(grid, index: int, sample: dict, cfg: Any, unit_sphere_points: np.ndarray = None):

    # Define incident field
    inc_dir = np.array(sample['source']['direction'])
    Ampl = sample['source']['amplitude']
    k = 2 * np.pi / sample['source']['wavelength']

    # Define space
    piecewise_const_space = bempp_cl.api.function_space(grid, "P", 1)

    # Defeine boundary operator

    identity = bempp_cl.api.operators.boundary.sparse.identity(
        piecewise_const_space, piecewise_const_space, piecewise_const_space
    )
    adlp = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
        piecewise_const_space, piecewise_const_space, piecewise_const_space, k
    )
    slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(
        piecewise_const_space, piecewise_const_space, piecewise_const_space, k
    )

    lhs = 0.5 * identity + adlp - 1j * k * slp
        
    @bempp_cl.api.complex_callable
    def combined_data(x, n, domain_index, result):
        result[0] = Ampl * 1j * k * np.exp(1j * k * np.sum(inc_dir * x)) * (np.sum(inc_dir * n) - 1) # Combined
        
    grid_fun = bempp_cl.api.GridFunction(piecewise_const_space, fun=combined_data)

    u_gamma, info, iteration_count = bempp_cl.api.linalg.gmres(lhs, -grid_fun, tol=cfg.gmres_tolerance, return_iteration_count=True)
    coefs = u_gamma.coefficients

    if cfg.generate_all_sol:

        # # Generate the far field solution
        # slp_ff = bempp_cl.api.operators.far_field.helmholtz.double_layer(piecewise_const_space, unit_sphere_points, k)

        # ff_sol = slp_ff.evaluate(u_gamma)

        # ff_sol_path = os.path.join(cfg.save_path, cfg.dir_name, f"far_field/{str(index).zfill(6)}")
        # np.save(ff_sol_path, ff_sol)

        # Space grid discretization for the visu solution
        Nx = cfg.num_point_full_sol
        Ny = cfg.num_point_full_sol
        xmin, xmax, ymin, ymax = 0, cfg.environment.size, 0, cfg.environment.size
        plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
        points = np.vstack((plot_grid[0].ravel(),
                            plot_grid[1].ravel(),
                            np.zeros(plot_grid[0].size)))
        u_evaluated = np.zeros(points.shape[1], dtype=np.complex128)
        u_evaluated[:] = np.nan

        obs_free_mask = ~get_obstacle_mask(points.T, sample['obstacles'], env_dim=3)
        obs_free_points = points[:, obs_free_mask]

        # Generate the potential solution on the grid for visu
        slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(piecewise_const_space, obs_free_points, k)

        free_field = Ampl * np.exp(1j * k * np.sum(inc_dir.reshape(-1, 1) * obs_free_points, axis=0))

        res = free_field + slp_pot.evaluate(u_gamma)
        u_evaluated[obs_free_mask] = res.flat

        full_sol_path = os.path.join(cfg.save_path, cfg.dir_name, f"full_sol/{str(index).zfill(6)}")
        np.save(full_sol_path, u_evaluated.reshape((Nx, Ny)).T)

        num_iter_path = os.path.join(cfg.save_path, cfg.dir_name, f"num_iter/{str(index).zfill(6)}")
        with open(num_iter_path, "w") as f:
            f.write(str(iteration_count))

    return np.vstack((coefs.real, coefs.imag)).T