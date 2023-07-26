def example_set_leaves_at_indices():
    import jax.numpy as jnp

    import cglib.tree_util

    pytree_to_modify = (jnp.array([0, 1, 2, 3]), jnp.array([True, False, True, False]))
    reference_pytree = (jnp.array([4, 5, 6, 7]), jnp.array([False, True, False, True]))
    indices = jnp.array([0, 2])
    print(cglib.tree_util.set_leaves_at_indices(pytree_to_modify, reference_pytree, indices))
    # (DeviceArray([4, 1, 6, 3], dtype=int32), DeviceArray([False, False, False, False], dtype=bool))


def example_all_equal():
    import jax.numpy as jnp

    import cglib.tree_util

    pytree_1 = (jnp.array([0, 1, 2, 3]), jnp.array([True, False, True, False]))
    pytree_2 = (jnp.array([4, 5, 6, 7]), jnp.array([False, True, False, True]))
    print(cglib.tree_util.all_equal(pytree_1, pytree_2))
    # False
    pytree_1 = (jnp.array([0, 1, 2, 3]), jnp.array([True, False, True, False]))
    pytree_2 = (jnp.array([0, 1, 2, 3]), jnp.array([True, False, True, False]))
    print(cglib.tree_util.all_equal(pytree_1, pytree_2))
    # True


if __name__ == "__main__":
    example_name = 'all_equal'

    match example_name:
        case 'set_leaves_at_indices':
            example_set_leaves_at_indices()
        case 'all_equal':
            example_all_equal()
        case other:
            print('No match found')