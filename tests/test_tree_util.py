def test_concatenate():
    import jax.numpy as jnp

    import cglib.tree_util

    my_pytree = []
    my_pytree.append((jnp.array([0, 1]), jnp.array([True, False])))
    my_pytree.append((jnp.array([2, 3]), jnp.array([False, True])))
    res = cglib.tree_util.concatenate(my_pytree)
    exp_res = (jnp.array([0, 1, 2, 3]), jnp.array([True, False, False, True]))
    assert cglib.tree_util.all_equal(res, exp_res)


def test_shape_dtype():
    import jax.numpy as jnp
    from jax import ShapeDtypeStruct

    import cglib.tree_util
    import cglib.type

    my_pytree = []
    my_pytree.append((jnp.array([0, 1]), jnp.array([True, False])))
    my_pytree.append((jnp.array([2, 3]), jnp.array([False, True])))
    res: list[tuple[ShapeDtypeStruct]] = cglib.tree_util.shape_dtype(my_pytree)
    res_bool = True
    for i in range(2):
        for j in range(2):
            res_bool = res_bool and res[i][j].shape[0] == 2
            if j == 0:
                res_bool = res_bool and res[i][j].dtype == cglib.type.int_
            else:
                res_bool = res_bool and res[i][j].dtype == jnp.bool_
    assert res_bool


def test_nbytes():
    import jax.numpy as jnp

    import cglib.tree_util

    my_pytree = []
    my_pytree.append((jnp.array([0, 1]), jnp.array([True, False])))
    my_pytree.append((jnp.array([2, 3]), jnp.array([False, True])))
    nbytes = cglib.tree_util.nbytes(my_pytree)
    assert nbytes == 20


def test_leaves_at_indices():
    import jax.numpy as jnp

    import cglib.tree_util

    my_pytree = []
    my_pytree.append((jnp.array([0, 1]), jnp.array([True, False])))
    my_pytree.append((jnp.array([2, 3]), jnp.array([False, True])))
    indices = jnp.array([1])
    res = cglib.tree_util.leaves_at_indices(indices, my_pytree)
    exp_res = [(jnp.array([1]), jnp.array([False])),
               (jnp.array([3]), jnp.array([True]))]
    assert cglib.tree_util.all_equal(res, exp_res)


def test_slice():
    import jax.numpy as jnp

    import cglib.tree_util

    my_pytree = []
    my_pytree.append((jnp.array([0, 1, 2, 3]), jnp.array([True, False, True, False])))
    my_pytree.append((jnp.array([4, 5, 6, 7]), jnp.array([False, True, False, True])))
    start = 1
    end = 3
    pytree_sliced = cglib.tree_util.slice(start, end, my_pytree)
    exp_slice = [(jnp.array([1, 2]), jnp.array([False, True])),
                 (jnp.array([5, 6]), jnp.array([True, False]))]
    assert cglib.tree_util.all_equal(pytree_sliced, exp_slice)


def test_set_leaves_at_indices():
    import jax.numpy as jnp

    import cglib.tree_util

    pytree_to_modify = (jnp.array([0, 1, 2, 3]), jnp.array([True, False, True, False]))
    reference_pytree = (jnp.array([4, 5, 6, 7]), jnp.array([False, True, False, True]))
    indices = jnp.array([0, 2])
    res = cglib.tree_util.set_leaves_at_indices(pytree_to_modify, reference_pytree, indices)
    exp_res = (jnp.array([4, 1, 6, 3]), jnp.array([False, False, False, False]))
    assert cglib.tree_util.all_equal(res, exp_res)
