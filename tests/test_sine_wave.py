import jax
import jax.lax
import jax.numpy as jnp

import cglib.grid
import cglib.grid.cell
import cglib.point_data
import cglib.sine_wave
import cglib.tree_util
import cglib.type


def helper_unpack_data():
    # Sine wave parameters
    p = jnp.array([2.0, 3.0])
    d_angle = jnp.pi * 0.25
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)])
    phase = jnp.array([jnp.pi])
    data = jnp.concatenate((d, phase))
    sine_wave = cglib.point_data.PointData(p, data)
    res = cglib.sine_wave.unpack_data(sine_wave)
    res_exp = (
        jnp.array([2.0, 3.0]),
        jnp.array([0.70710677, 0.7071068]),
        jnp.array(3.1415927),
    )
    return cglib.tree_util.all_isclose(res, res_exp)


def test_unpack_data():
    assert jax.jit(helper_unpack_data)()


def helper_change_phase_to_align_i_with_j_1d():
    p_i = 0.0
    phase_j = jnp.pi
    f = 1.0
    inv_d = True
    res = cglib.sine_wave.change_phase_to_align_i_with_j_1d(p_i, phase_j, f, inv_d)
    res_exp = 0.0
    return res == res_exp


def test_change_phase_to_align_i_with_j_1d():
    assert jax.jit(helper_change_phase_to_align_i_with_j_1d)()


def helper_change_phase_to_align_i_with_j_nd():
    p_i = jnp.array([0.0, 1.0])
    d_angle_i = jnp.pi
    d_i = jnp.array([jnp.cos(d_angle_i), jnp.sin(d_angle_i)])
    p_j = jnp.array([1.0, 0.0])
    d_angle_j = 0.1
    d_j = jnp.array([jnp.cos(d_angle_j), jnp.sin(d_angle_j)])
    phase_j = jnp.array([jnp.pi])
    data_j = jnp.concatenate((d_j, phase_j))
    sine_wave_j = cglib.point_data.PointData(p_j, data_j)
    f = 1.0
    phase_i = cglib.sine_wave.change_phase_to_align_i_with_j_nd(
        sine_wave_j, p_i, d_i, f
    )
    phase_i_exp = 5.6245236
    return jnp.isclose(phase_i, phase_i_exp)


def test_change_phase_to_align_i_with_j_nd():
    assert jax.jit(helper_change_phase_to_align_i_with_j_nd)()


def helper_prolong_j_into_i():
    p_i = jnp.array([0.0, 1.0])
    d_angle_i = jnp.pi
    d_i = jnp.array([jnp.cos(d_angle_i), jnp.sin(d_angle_i)])
    phase_i = jnp.array([jnp.pi])
    constraint_i = jnp.array([0.0])
    p_j = jnp.array([1.0, 0.0])
    d_angle_j = 0.0
    d_j = jnp.array([jnp.cos(d_angle_j), jnp.sin(d_angle_j)])
    phase_j = jnp.array([jnp.pi])
    constraint_j = jnp.array([0.0])
    data_i = jnp.concatenate((d_i, phase_i, constraint_i))
    sine_wave_i = cglib.point_data.PointData(p_i, data_i)
    data_j = jnp.concatenate((d_j, phase_j, constraint_j))
    sine_wave_j = cglib.point_data.PointData(p_j, data_j)
    f = 0.5
    res = cglib.sine_wave.prolong_j_into_i(sine_wave_j, sine_wave_i, f)
    res_exp = cglib.point_data.PointData(
        point=jnp.array([0.0, 1.0]), data=jnp.array([1.0, 0.0, 0.0, 0.0])
    )
    return cglib.tree_util.all_isclose(res, res_exp)


def test_prolong_j_into_i():
    assert jax.jit(helper_prolong_j_into_i)()


def helper_phase_alignment_energy_i_with_j_1d():
    p_i = 1.0
    phase_i = jnp.pi
    phase_j = jnp.pi
    f = 1.0
    inv_d = True
    res = cglib.sine_wave.phase_alignment_energy_i_with_j_1d(
        p_i, phase_i, phase_j, f, inv_d
    )
    return res == 1.0


def test_phase_alignment_energy_i_with_j_1d():
    assert jax.jit(helper_phase_alignment_energy_i_with_j_1d)()


def helper_phase_alignment_energy_i_with_j_nd():
    p_i = jnp.array([0.0, 1.0])
    d_angle_i = jnp.pi
    d_i = jnp.array([jnp.cos(d_angle_i), jnp.sin(d_angle_i)])
    phase_i = jnp.array([jnp.pi])
    p_j = jnp.array([1.0, 0.0])
    d_angle_j = 0.0
    d_j = jnp.array([jnp.cos(d_angle_j), jnp.sin(d_angle_j)])
    phase_j = jnp.array([jnp.pi])
    data_i = jnp.concatenate((d_i, phase_i))
    sine_wave_i = cglib.point_data.PointData(p_i, data_i)
    data_j = jnp.concatenate((d_j, phase_j))
    sine_wave_j = cglib.point_data.PointData(p_j, data_j)
    f = 0.5
    alignment_energy = cglib.sine_wave.phase_alignment_energy_i_with_j_nd(
        sine_wave_i, sine_wave_j, f
    )
    return alignment_energy == 0.0


def test_phase_alignment_energy_i_with_j_nd():
    assert jax.jit(helper_phase_alignment_energy_i_with_j_nd)()


def helper_grid_align():
    with jax.ensure_compile_time_eval():
        # Each sine wave will be associated with a unique grid cell.
        grid_cell_2dcount = jnp.array([2, 3])
        grid_origin = jnp.array([-1.0, 2.0])
        grid_cell_sides_length = 0.5
        grid = cglib.grid.Grid(grid_cell_2dcount, grid_origin, grid_cell_sides_length)
        iter_count = 25
        mode = 0
        sine_wave_count = 6
    seed = 1701
    seed_jax = jax.random.PRNGKey(seed)
    seed_jax = jax.random.split(seed_jax, 4)
    # Random sine waves' parameters
    p = cglib.grid.cell.center_points_jittered(grid, seed_jax[0], 1.0)
    d_angle = jax.random.uniform(
        seed_jax[1], (sine_wave_count,), cglib.type.float_, -jnp.pi, jnp.pi
    )
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)]).T
    phase = jax.random.uniform(
        seed_jax[2], (sine_wave_count,), cglib.type.float_, -jnp.pi, jnp.pi
    )
    constraint = jnp.zeros((sine_wave_count,))
    f = 1.0 / (0.2 * grid.cell_sides_length)
    data = jnp.concatenate(
        (d, phase.reshape(sine_wave_count, 1), constraint.reshape(sine_wave_count, 1)),
        axis=1,
    )
    sine_waves = cglib.point_data.PointData(p, data)
    grid_sine_waves = cglib.point_data.GridPointData(sine_waves, grid)
    use_spatial_weights = True

    energy = jnp.empty((iter_count,))

    for i in range(iter_count):
        grid_sine_wave_energy = cglib.sine_wave.grid_alignment_energy(
            grid_sine_waves, f, use_spatial_weights, mode
        )
        energy.at[i].set(jnp.average(grid_sine_wave_energy))
        grid_sine_waves = cglib.sine_wave.grid_align(
            grid_sine_waves, f, use_spatial_weights
        )

    return energy[-1] < 0.01


def test_grid_align():
    assert jax.jit(helper_grid_align)()


def helper_eval_angle():
    # Evaluation point
    x = jnp.array([0.0, 1.0])
    # Sine wave parameters
    p = jnp.array([2.0, 3.0])
    d_angle = jnp.pi * 0.25
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)])
    phase = jnp.pi
    f = 0.5
    # Sine wave angle
    res = cglib.sine_wave.eval_angle(x, p, d, phase, f)
    res_exp = -5.744174
    return jnp.isclose(res, res_exp)


def test_eval_angle():
    assert jax.jit(helper_eval_angle)()


def helper_eval_complex_angle():
    # Evaluation point
    x = jnp.array([0.0, 1.0])
    # Sine wave parameters
    p = jnp.array([2.0, 3.0])
    d_angle = jnp.pi * 0.25
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)])
    phase = jnp.pi
    f = 0.5
    # Sine wave angle
    res = cglib.sine_wave.eval_complex_angle(x, p, d, phase, f)
    res_exp = jnp.array(0.8582166 + 0.5132877j)
    return jnp.isclose(res, res_exp)


def test_eval_complex_angle():
    assert jax.jit(helper_eval_complex_angle)()


def helper_eval():
    # Evaluation point
    x = jnp.array([0.0, 1.0])
    # Sine wave parameters
    p = jnp.array([2.0, 3.0])
    d_angle = jnp.pi * 0.25
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)])
    phase = jnp.pi
    f = 0.5
    # Sine wave angle
    res = cglib.sine_wave.eval(x, p, d, phase, f)
    res_exp = 0.5132877
    return jnp.isclose(res, res_exp)


def test_eval():
    assert jax.jit(helper_eval)()
