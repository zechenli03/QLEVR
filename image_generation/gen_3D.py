import json
import os
import argparse
import sys
from datetime import datetime as dt
import random

# import fractions

# import pydevd
#
# pydevd.settrace('127.0.0.1', port=1090, stdoutToServer=True, stderrToServer=True)

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

# The 2D bounding box generation code is taken from:
# https://blender.stackexchange.com/questions/7198/save-the-2d-bounding-box-of-an-object-in-rendered-image-to-a-text-file

INSIDE_BLENDER = True
try:
    import bpy
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils_3d as utils
        from mathutils import Vector
    except ImportError as e:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils_3d.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.92).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='./data/Backdrop.blend',
                    help="Base blender file on which all scenes are based; includes " +
                         "ground plane, lights, camera, and denoising setting.")
parser.add_argument('--properties_json', default='./data/properties.json',
                    help="JSON file defining objects, materials, sizes, and colors. " +
                         "The \"colors\" field maps from 2d color names to RGB values; " +
                         "The \"sizes\" field maps from 2d size names to scalars used to " +
                         "rescale object models; the \"materials\" and \"shapes\" fields map " +
                         "from material and shape names to .blend files in the " +
                         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='./data/shapes',
                    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='./data/materials',
                    help="Directory where .blend files for materials are stored")
parser.add_argument('--input_dir', default='./output/2d_scene/')
parser.add_argument('--plane_img_dir', default='images')
parser.add_argument('--plane_mat_dir', default='images_materials')
parser.add_argument('--filename_2d_prefix', default='2d')
parser.add_argument('--filename_mat_prefix', default='blender')
parser.add_argument('--json_2d_prefix', default='scenes_2d')
parser.add_argument('--split', default='train')
parser.add_argument('--ratio_of_2d_to_3d', default=100, type=int)
parser.add_argument('--min_pixels_per_object', default=100, type=int,
                    help="All objects will have at least this many visible pixels in the " +
                         "final rendered images; this ensures that no objects are fully " +
                         "occluded by other objects.")

# Json Info
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
                    help="String to store in the \"date\" field of the generated JSON file;"
                         "defaults to today's date")
parser.add_argument('--version', default='1.0',
                    help="String to store in the \"version\" field of the generated JSON file")

# Output settings
parser.add_argument('--start_idx', default=None, type=int,
                    help="The index at which to start for numbering rendered images. Setting " +
                         "this to non-zero values allows you to distribute rendering across " +
                         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=20, type=int,
                    help="The number of images to render")
parser.add_argument('--filename_3d_prefix', default='3d',
                    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--json_3d_prefix', default='scenes_3d')
parser.add_argument('--output_dir', default='./output/3d_scene/',
                    help="The directory where output images will be stored. It will be " +
                         "created if it does not exist.")
parser.add_argument('--output_image_dir', default='images',
                    help="The directory where output images will be stored. It will be " +
                         "created if it does not exist.")
parser.add_argument('--output_blend_dir', default='blend_files',
                    help="The directory where blender scene files will be stored, if the " +
                         "user requested that these files be saved using the " +
                         "--save_blendfiles flag; in this case it will be created if it does " +
                         "not already exist.")
parser.add_argument('--save_blendfiles', action='store_true', default=False,
                    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
                         "each generated image to be stored in the directory specified by " +
                         "the --output_blend_dir flag. These files are not saved by default " +
                         "because they take up ~5-10MB each.")

# Rendering options
parser.add_argument('--use_gpu', action='store_true', default=False,
                    help="Setting --use_gpu enables GPU-accelerated rendering using CUDA. " +
                         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
                         "to work.")
parser.add_argument('--width', default=960, type=int,
                    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=720, type=int,
                    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=0.1, type=float,
                    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=0.1, type=float,
                    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=0.1, type=float,
                    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--light_rtt_jitter', default=2, type=float,
                    help="The rotation of random jitter to add to the back light position.")
parser.add_argument('--light_power_jitter', default=10.0, type=float,
                    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_loc_jitter', default=0.1, type=float,
                    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--camera_rtt_jitter', default=0.3, type=float,
                    help="The rotation of random jitter to add to the camera position")
parser.add_argument('--camera_position', default=[[2, -3, 5.5], [2, -4.2, 4.5], [2, -4.8, 3.5]], type=list,
                    help="The camera position")
parser.add_argument('--camera_rotation', default=[[40, 0, 0], [51, 0, 0], [62, 0, 0]], type=list,
                    help="The camera rotation")
parser.add_argument('--render_num_samples', default=256, type=int,
                    help="The number of samples to use when rendering. Larger values will " +
                         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_transparent_min_bounces', default=8, type=int,
                    help="The minimum number of transparent bounces to use for rendering.")
parser.add_argument('--render_transparent_max_bounces', default=8, type=int,
                    help="The maximum number of transparent bounces to use for rendering.")
parser.add_argument('--render_transmission_bounces', default=16, type=int,
                    help="The maximum number of transmission bounces to use for rendering.")
parser.add_argument('--render_total_max_bounces', default=16, type=int,
                    help="The total maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=16, type=int,
                    help="The tile size to use for rendering. This should not affect the " +
                         "quality of the rendered image but may affect the speed; CPU-based " +
                         "rendering may achieve better performance using smaller tile sizes " +
                         "while larger tile sizes may be optimal for GPU-based rendering.")


def render_3d_scene(cam_pos=None):
    scene_3d = {'image_index': i + start_idx,
                'planes': [],
                'plane_spatial_relations': {},
                'plane_pixel_relations': {
                    'behind': [],
                    'front': [],
                    'left': [],
                    'right': []
                },
                'objects': [],
                'obj_spatial_relations': {},
                'obj_pixel_relations': {
                    'behind': [],
                    'front': [],
                    'left': [],
                    'right': []
                },
                'blender_info': {
                    'directions': {},
                }
                }

    # load backdrop base scene blender file
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments
    utils.set_render(output_img_path, args)

    # Set camera arguments
    cam_index = utils.set_camera(args, scene_3d, cam_pos)

    camera = bpy.data.objects['Camera']

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(size=20)
    temp_plane = bpy.context.object

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    plane_normal = temp_plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    bpy.data.objects.remove(temp_plane, do_unlink=True)

    # Save all six axis-aligned directions in the scene struct
    scene_3d['blender_info']['directions']['behind'] = tuple(plane_behind)
    scene_3d['blender_info']['directions']['front'] = tuple(-plane_behind)
    scene_3d['blender_info']['directions']['left'] = tuple(plane_left)
    scene_3d['blender_info']['directions']['right'] = tuple(-plane_left)
    scene_3d['blender_info']['directions']['above'] = tuple(plane_up)
    scene_3d['blender_info']['directions']['below'] = tuple(-plane_up)

    # Set lighting arguments
    utils.set_light(args)

    # Create a plane on the ground,
    # then put the 2D image on it as floor
    # Get the plane size
    width = properties["plane_size"]["width"]
    length = properties["plane_size"]["length"]
    # f = fractions.Fraction(width, length)
    # img_ratio = int(width / f.numerator)
    utils.create_plane(length, width, img_plane_path, grayscale_img_path)

    # Add the objects based on the json information file
    # Snap the objects to the plane
    bpy.context.scene.tool_settings.snap_elements = {'FACE'}
    bpy.context.scene.tool_settings.use_snap = True
    scene_3d['objects'], blender_objects = utils.add_objects(scene[scene_index], properties, camera, args)

    if cam_index != 0:
        # Check that all objects are at least partially visible in the rendered image
        all_visible = utils.check_visibility(blender_objects, args.min_pixels_per_object, args.use_gpu)
        if not all_visible:
            # If any of the objects are fully occluded then start over; delete all
            # objects from the scene and place them all again.
            print(' ****************** Some objects are occluded; changing camera position ****************** ')
            return render_3d_scene(cam_pos=random.randint(0, cam_index - 1))

    for p in scene[scene_index]['planes']:
        cc = [200, 200] if type(p['center_coordinate']) is not list else p['center_coordinate'].copy()
        cc.append(0.0)
        cc = Vector([x / args.ratio_of_2d_to_3d for x in cc])
        p_bound_box = utils.plane_bounding_box(p, camera, args.ratio_of_2d_to_3d)
        scene_3d['planes'].append({
            'shape': p['shape'],
            'material': p['material'],
            'color': p['color'],
            'internal_objs': p['internal_objs'],
            'center_pixel_coords': utils.get_camera_coords(camera, cc),
            'bounding_box': {
                'left': p_bound_box[0],
                'top': p_bound_box[1],
                'width': p_bound_box[2],
                'height': p_bound_box[3]
            }
        })

    # eps_obj = properties['sizes']['small']/(2*args.ratio_of_2d_to_3d)
    utils.compute_spatial_relationships(scene_3d, scene[scene_index], args.ratio_of_2d_to_3d)
    utils.compute_pixel_relationships(scene_3d)

    # Render the scene
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as ect:
            print(ect)

    # Save blend file
    if output_blend_path is not None:
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)

    # Record image information into a json file
    json_dic['scenes'].append(scene_3d)
    with open(json_3d_path, 'w', encoding='utf-8') as file:
        # json.dump(json_dic, file, indent=1)
        json.dump(json_dic, file)
    print(f'Save picture and json file {i + start_idx}')


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)

        # Input info
        num_digits = 6
        input_path = os.path.join(args.input_dir, args.split)

        img2d_path = os.path.join(input_path, args.plane_img_dir)
        assert os.path.exists(img2d_path)
        img2d_prefix = '%s_%s_' % (args.filename_2d_prefix, args.split)
        img2d_template = '%s%%0%dd.png' % (img2d_prefix, num_digits)
        img2d_template = os.path.join(img2d_path, img2d_template)

        grayscale_img_path = os.path.join(input_path, args.plane_mat_dir)
        assert os.path.exists(grayscale_img_path)
        grayscale_prefix = '%s_%s_' % (args.filename_mat_prefix, args.split)
        grayscale_img_template = '%s%%0%dd.png' % (grayscale_prefix, num_digits)
        grayscale_img_template = os.path.join(grayscale_img_path, grayscale_img_template)

        json_2d_prefix = '%s_%s.json' % (args.json_2d_prefix, args.split)
        json_2d_path = os.path.join(input_path, json_2d_prefix)
        assert os.path.exists(json_2d_path)

        # Output Info
        output_path = os.path.join(args.output_dir, args.split)

        img3d_path = os.path.join(output_path, args.output_image_dir)
        img3d_prefix = '%s_%s_' % (args.filename_3d_prefix, args.split)
        img3d_template = '%s%%0%dd.png' % (img3d_prefix, num_digits)
        img3d_template = os.path.join(img3d_path, img3d_template)

        output_blend_path = os.path.join(output_path, args.output_blend_dir)
        blend_template = '%s%%0%dd.blend' % (img3d_prefix, num_digits)
        blend_template = os.path.join(output_blend_path, blend_template)

        json_3d_prefix = '%s_%s.json' % (args.json_3d_prefix, args.split)
        json_3d_path = os.path.join(output_path, json_3d_prefix)

        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        if not os.path.exists(img3d_path):
            os.makedirs(img3d_path)
        if args.save_blendfiles and not os.path.isdir(output_blend_path):
            os.makedirs(output_blend_path)
        if not os.path.exists(json_3d_path):
            with open(json_3d_path, 'w', encoding='utf-8') as _:
                json_dic = {
                    'info': {
                        'date': args.date,
                        'version': args.version,
                        'split': args.split
                    },
                    'scenes': []
                }
        else:
            with open(json_3d_path, 'r', encoding='utf-8') as f:
                json_dic = json.load(f)
            if args.start_idx is not None:
                if json_dic['scenes'][-1]['image_index'] >= args.start_idx:
                    assert json_dic['scenes'][
                               args.start_idx - (json_dic['scenes'][-1]['image_index'] - len(json_dic['scenes']) + 1)][
                               'image_index'] == args.start_idx, \
                        "The index of the json information is conflicts with the start index of the image to be " \
                        "generated. "
                    print(
                        "* The image and json information starting with the 'start_idx' already exist in the "
                        "directory.")
                    print(" * Do you want to overwrite this information and continue to generate? (y/n)")
                    in_content = input("Input：")
                    if in_content == "y":
                        json_dic['scenes'] = json_dic['scenes'][:args.start_idx - (
                                json_dic['scenes'][-1]['image_index'] - len(json_dic['scenes']) + 1)]
                        if os.path.exists(output_blend_path):
                            p = max([os.listdir(output_blend_path), os.listdir(img3d_path)], key=len)
                        else:
                            p = os.listdir(img3d_path)
                        for filename in p:
                            index = filename.split('_')[-1]
                            index = int(index.split('.')[0])
                            if index >= args.start_idx:
                                if os.path.exists(img3d_template % index): os.remove(img3d_template % index)
                                if os.path.exists(blend_template % index): os.remove(blend_template % index)
                    else:
                        sys.exit()
                else:
                    assert json_dic['scenes'][-1]['image_index'] == args.start_idx - 1, \
                        "The 'start_idx' is not close to the previous one. Please confirm whether the 'start_idx' is " \
                        "correct. "

        # Load 2d scene json file
        with open(json_2d_path) as f:
            data = json.load(f)
        scene = data['scenes']

        with open(args.properties_json, 'r') as f:
            properties = json.load(f)

        # Make sure 2D image directory is not empty
        assert len(os.listdir(img2d_path)) != 0

        if args.start_idx is not None:
            start_idx = args.start_idx
        else:
            start_idx = 0
            if os.listdir(output_path):
                # This directory is not empty
                for filename in os.listdir(img3d_path):
                    index = filename.split('_')[-1]
                    index = int(index.split('.')[0])
                    start_idx = index + 1 if index >= start_idx else start_idx

        if args.num_images is not None:
            num_images = args.num_images
        else:
            num_images = len(os.listdir(img2d_path))

        for i in range(num_images):
            img_plane_path = img2d_template % (i + start_idx)
            grayscale_img_path = grayscale_img_template % (i + start_idx)
            output_img_path = img3d_template % (i + start_idx)
            output_blend_path = None
            if args.save_blendfiles:
                output_blend_path = blend_template % (i + start_idx)

            # Make sure this 2D image exists
            assert os.path.exists(img_plane_path)
            assert os.path.exists(grayscale_img_path)

            # Make sure scenes.json has this 2D image's information
            scene_index = i + start_idx - (scene[-1]['image_index'] - len(scene) + 1)
            assert scene[scene_index]['image_index'] == i + start_idx

            # Start rendering 3d images
            render_3d_scene()
            print("*************************************************************************************")

    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python ./image_generation/gen_3d.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')
