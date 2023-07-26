import filecmp

import jax
import jax.numpy as jnp

import cglib.texture
import cglib.tree_util


def test_twilight_from_png_1channel_8bits():
    filename_in = 'data/png/stress.png'
    filename_out = 'data/png/stress_twilight.png'
    filename_out_ref = 'data/png/stress_twilight_ref.png'
    clamp = cglib.texture.ImageWrap.CLAMP
    device = jax.devices('cpu')[0]

    cglib.texture.twilight_from_png_1channel_8bits(
        filename_in, filename_out, device)

    im_tex = cglib.texture.image_create_from_png_1channel_8bits(
        filename_out, clamp, device)
    im_tex_ref = cglib.texture.image_create_from_png_1channel_8bits(
        filename_out_ref, clamp, device)
    assert cglib.tree_util.all_isclose(im_tex, im_tex_ref)


def test_image_png_from_hamiltonian_format():
    output_from_HCode_filename = \
        'data/hamiltonian_cycle/ani_to_iso_dir_field_output_from_HCode.txt'

    dir_field_filename = 'data/png/H_ani_to_iso_dir_field_template.png'
    dir_field_mask_filename = 'data/png/H_ani_to_iso_dir_field_mask.png'
    cglib.texture.image_png_from_hamiltonian_format(
        output_from_HCode_filename,
        dir_field_filename,
        dir_field_mask_filename)

    clamp = cglib.texture.ImageWrap.CLAMP
    device = jax.devices('cpu')[0]
    tex = cglib.texture.image_create_from_png_1channel_8bits(
        dir_field_filename, clamp, device)
    tex_mask = cglib.texture.image_create_from_png_1channel_8bits(
        dir_field_mask_filename, clamp, device)

    dir_field_filename_ref = 'data/png/H_ani_to_iso_dir_field_template_ref.png'
    dir_field_mask_filename_ref = \
        'data/png/H_ani_to_iso_dir_field_mask_ref.png'
    tex_ref = cglib.texture.image_create_from_png_1channel_8bits(
        dir_field_filename_ref, clamp, device)
    tex_mask_ref = cglib.texture.image_create_from_png_1channel_8bits(
        dir_field_mask_filename_ref, clamp, device)
    res0 = cglib.tree_util.all_isclose(tex_ref, tex)
    res1 = cglib.tree_util.all_isclose(tex_mask_ref, tex_mask)
    assert jnp.logical_and(res0, res1)


def test_image_hamiltonian_format_from_png():
    dir_field_filename = 'data/png/H_ani_to_iso_dir_field.png'
    dir_field_mask_filename = 'data/png/H_ani_to_iso_dir_field_mask.png'

    input_for_HCode = \
        'data/hamiltonian_cycle/ani_to_iso_dir_field_input_for_HCode.txt'
    cglib.texture.image_hamiltonian_format_from_png(
        dir_field_filename, dir_field_mask_filename, input_for_HCode)

    input_for_HCode_ref = \
        'data/hamiltonian_cycle/ani_to_iso_dir_field_input_for_HCode_ref.txt'

    # reading files
    f1 = open(input_for_HCode, "r")
    f2 = open(input_for_HCode_ref, "r")

    f1_data = f1.readlines()
    f2_data = f2.readlines()

    identical = True
    for i in range(len(f1_data)):
        f1_line_i = f1_data[i]
        f2_line_i = f2_data[i]

        if f1_line_i != f2_line_i:
            identical = False
            break

    # closing files
    f1.close()
    f2.close()

    assert identical
