import argparse

import jax
import numpy as np
import jax.numpy as jnp

import cglib.backend
import cglib.direction
import cglib.fdm_aa
import cglib.grid
import cglib.point_data

if __name__ == "__main__":

    device_cpu, device_gpu = cglib.backend.get_cpu_and_gpu_devices()

    parser = argparse.ArgumentParser(
        description='Input: A json file containing all the input parameters. Output: '
    )
    parser.add_argument(
        "input_filename", help="Use the inputs in the file specified here.")
    parser.add_argument(
        "output_filename", help="Filename of the output")
    args = parser.parse_args()

    input_param_filename = args.input_filename
    fname = args.output_filename

    parameters = cglib.fdm_aa.Parameters()
    parameters.load(input_param_filename)

    # Get the grid
    grid_sines_aligned = cglib.point_data.grid_load(
        parameters.grid_sines_aligned_filename)
    cell_ndcount = grid_sines_aligned.grid.cell_ndcount
    cell_ndcount = np.concatenate((cell_ndcount, np.array([1])))
    origin = grid_sines_aligned.grid.origin
    origin = np.concatenate((origin, np.array([0.])))
    grid_sines_aligned_grid = cglib.grid.Grid(
        cell_ndcount, origin, grid_sines_aligned.grid.cell_sides_length)
    
    phases = jax.device_put(
        grid_sines_aligned.point_data.data[:, 2], device_cpu)
    phases = jnp.reshape(phases, (-1, 1))
    mask = jnp.isnan(phases)
    # Get cartesian 2D sine directions
    direction_cartesian = jax.device_put(
        grid_sines_aligned.point_data.data[:, :2], device_cpu)
    direction_polar = jax.vmap(cglib.direction.cartesian_to_polar)(
        direction_cartesian)
    direction_polar = jnp.reshape(direction_polar, (-1, 1))
    pi_over_2 = jnp.full((direction_polar.shape[0], 1), jnp.pi*0.5)
    pi_over_2 = jnp.where(mask, jnp.nan, pi_over_2)
    data = jnp.concatenate(
        (pi_over_2, direction_polar, phases),
        axis=1)

    # Save the grid
    cglib.grid.savetxt(fname, grid_sines_aligned_grid)
    with open(fname, 'a') as f:
        np.savetxt(f, data, fmt='%.4f')