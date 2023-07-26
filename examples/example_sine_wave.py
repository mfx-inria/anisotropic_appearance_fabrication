import jax
import jax.numpy as jnp

import cglib.grid
import cglib.grid.cell
import cglib.point_data
import cglib.sine_wave
import cglib.type


def helper_grid_align_with_neighbors():
    with jax.ensure_compile_time_eval():
        # Each sine wave will be associated with a unique grid cell.
        grid_cell_2dcount = jnp.array([2, 3])
        grid_origin = jnp.array([-1., 2.])
        grid_cell_sides_length = 0.5
        grid = cglib.grid.Grid(
            grid_cell_2dcount, grid_origin, grid_cell_sides_length)
    seed = 1701
    seed_jax = jax.random.PRNGKey(seed)
    seed_jax = jax.random.split(seed_jax, 4)
    # Random sine waves' parameters
    sine_wave_count = 6
    p = cglib.grid.cell.center_points_jittered(grid, seed_jax[0], 1.)
    d_angle = jax.random.uniform(
        seed_jax[1],
        (sine_wave_count,),
        cglib.type.float_,
        -jnp.pi, jnp.pi)
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)]).T
    phase = jax.random.uniform(
        seed_jax[2],
        (sine_wave_count,),
        cglib.type.float_,
        -jnp.pi, jnp.pi)
    f = 1. / (0.2 * grid.cell_sides_length)
    data = jnp.concatenate((d, phase.reshape(sine_wave_count, 1)), axis=1)
    sine_waves = cglib.point_data.PointData(p, data)
    grid_sine_waves = cglib.point_data.GridPointData(sine_waves, grid)
    i = 0
    use_spatial_weights = True
    res = cglib.sine_wave.grid_align_with_neighbors(
        i,
        grid_sine_waves,
        f,
        use_spatial_weights)
    return res


def example_grid_align_with_neighbors():
    print(jax.jit(helper_grid_align_with_neighbors)())


def example_grid_prolong_cell():
    with jax.ensure_compile_time_eval():
        # Each sine wave will be associated with a unique grid cell.
        grid_side_cell_count = 2
        grid_cell_2dcount = jnp.array(
            [grid_side_cell_count, grid_side_cell_count])
        grid_origin = jnp.array([-1., 2.])
        grid_cell_sides_length = 0.5
        grid = cglib.grid.Grid(
            grid_cell_2dcount, grid_origin, grid_cell_sides_length)
    seed = 1701
    seed_jax = jax.random.PRNGKey(seed)
    seed_jax = jax.random.split(seed_jax, 4)
    # Random sine waves' parameters
    sine_wave_count = grid_side_cell_count * grid_side_cell_count
    p = cglib.grid.cell.center_points_jittered(grid, seed_jax[0], 1.)
    d_angle = jax.random.uniform(
        seed_jax[1],
        (sine_wave_count,),
        cglib.type.float_,
        -jnp.pi, jnp.pi)
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)]).T
    phase = jax.random.uniform(
        seed_jax[2],
        (sine_wave_count,),
        cglib.type.float_,
        -jnp.pi, jnp.pi)
    constraint: jnp.ndarray = jnp.zeros((sine_wave_count,))
    f = 1. / (0.2 * grid.cell_sides_length)
    data = jnp.concatenate((d, phase.reshape(
        sine_wave_count, 1), constraint.reshape(sine_wave_count, 1)), axis=1)
    sine_waves = cglib.point_data.PointData(p, data)
    grid_sine_waves = cglib.point_data.GridPointData(sine_waves, grid)
    multigrid_sine_wave = cglib.sine_wave.multigrid_create(
        grid_side_cell_count,
        grid_sine_waves,
        f)
    grid_level_N_cell_ndindex = jnp.array([0, 0])
    sine_wave_prolong = cglib.sine_wave.grid_prolong_cell(
        grid_level_N_cell_ndindex,
        multigrid_sine_wave[1],
        multigrid_sine_wave[0],
        f)
    print("This sine wave")
    print(multigrid_sine_wave[1])
    print("will be prolonged into these sine waves")
    print(multigrid_sine_wave[0])
    print("Here the result")
    print(sine_wave_prolong)


def example_grid_prolong():
    with jax.ensure_compile_time_eval():
        # Each sine wave will be associated with a unique grid cell.
        grid_side_cell_count = 2
        grid_cell_2dcount = jnp.array(
            [grid_side_cell_count, grid_side_cell_count])
        grid_origin = jnp.array([-1., 2.])
        grid_cell_sides_length = 0.5
        grid = cglib.grid.Grid(
            grid_cell_2dcount, grid_origin, grid_cell_sides_length)
    seed = 1701
    seed_jax = jax.random.PRNGKey(seed)
    seed_jax = jax.random.split(seed_jax, 4)
    # Random sine waves' parameters
    sine_wave_count = grid_side_cell_count * grid_side_cell_count
    p = cglib.grid.cell.center_points_jittered(grid, seed_jax[0], 1.)
    d_angle = jax.random.uniform(
        seed_jax[1],
        (sine_wave_count,),
        cglib.type.float_,
        -jnp.pi, jnp.pi)
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)]).T
    phase = jax.random.uniform(
        seed_jax[2],
        (sine_wave_count,),
        cglib.type.float_,
        -jnp.pi, jnp.pi)
    constraint: jnp.ndarray = jnp.zeros((sine_wave_count,))
    f = 1. / (0.2 * grid.cell_sides_length)
    data = jnp.concatenate((d, phase.reshape(
        sine_wave_count, 1), constraint.reshape(sine_wave_count, 1)), axis=1)
    sine_waves = cglib.point_data.PointData(p, data)
    grid_sine_waves = cglib.point_data.GridPointData(sine_waves, grid)
    multigrid_sine_wave = cglib.sine_wave.multigrid_create(
        grid_side_cell_count,
        grid_sine_waves,
        f)
    sine_wave_prolong = cglib.sine_wave.grid_prolong(
        1,
        multigrid_sine_wave[1],
        multigrid_sine_wave[0],
        f)
    print("This sine wave")
    print(multigrid_sine_wave[1])
    print("is prolonged into these sine waves")
    print(multigrid_sine_wave[0])
    print("Here the result")
    print(sine_wave_prolong)


def helper_multigrid_create():
    with jax.ensure_compile_time_eval():
        # Each sine wave will be associated with a unique grid cell.
        grid_side_cell_count = 2
        grid_cell_2dcount = jnp.array(
            [grid_side_cell_count, grid_side_cell_count])
        grid_origin = jnp.array([-1., 2.])
        grid_cell_sides_length = 0.5
        grid = cglib.grid.Grid(
            grid_cell_2dcount, grid_origin, grid_cell_sides_length)
    seed = 1701
    seed_jax = jax.random.PRNGKey(seed)
    seed_jax = jax.random.split(seed_jax, 4)
    # Random sine waves' parameters
    sine_wave_count = grid_side_cell_count * grid_side_cell_count
    p = cglib.grid.cell.center_points_jittered(grid, seed_jax[0], 1.)
    d_angle = jax.random.uniform(
        seed_jax[1],
        (sine_wave_count,),
        cglib.type.float_,
        -jnp.pi, jnp.pi)
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)]).T
    phase = jax.random.uniform(
        seed_jax[2],
        (sine_wave_count,),
        cglib.type.float_,
        -jnp.pi, jnp.pi)
    constraint: jnp.ndarray = jnp.zeros((sine_wave_count,))
    f = 1. / (0.2 * grid.cell_sides_length)
    data = jnp.concatenate((d, phase.reshape(
        sine_wave_count, 1), constraint.reshape(sine_wave_count, 1)), axis=1)
    sine_waves = cglib.point_data.PointData(p, data)
    grid_sine_waves = cglib.point_data.GridPointData(sine_waves, grid)
    return cglib.sine_wave.multigrid_create(
        grid_side_cell_count,
        grid_sine_waves,
        f)


def example_multigrid_create():
    print(jax.jit(helper_multigrid_create)())


if __name__ == "__main__":
    example_name = 'multigrid_create'

    match example_name:
        case 'grid_align_with_neighbors':
            example_grid_align_with_neighbors()
        case 'grid_prolong_cell':
            example_grid_prolong_cell()
        case 'grid_prolong':
            example_grid_prolong()
        case 'multigrid_create':
            example_multigrid_create()
        case other:
            print('No match found')
