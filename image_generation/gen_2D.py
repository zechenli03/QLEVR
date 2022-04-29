import json
import os, sys
import random
import utils_2d as utils
import argparse
from datetime import datetime as dt
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

# Dir Info
parser.add_argument('--output_dir', default='./output/2d_scene/')
parser.add_argument('--split', default='train')
parser.add_argument('--img_dir', default='images')
parser.add_argument('--img_prefix', default='2d')
parser.add_argument('--full_img_dir', default='full_images')
parser.add_argument('--full_img_prefix', default='2d_full')
parser.add_argument('--img_mat_dir', default='images_materials')
parser.add_argument('--img_mat_prefix', default='blender')
parser.add_argument('--json_prefix', default='scenes_2d')
parser.add_argument('--plane_mat_col_dir', default='./data/plane_materials_colors',
                    help='Contains assets from ambientCG.com, licensed under CC0 1.0 Universal.')
parser.add_argument('--properties_json', default='./data/properties.json',
                    help='Record the plane and object information in the image, '
                         'which can be used to generate 2d and 3d pictures')

# Image Info
parser.add_argument('--start_idx', default=None, type=int)
parser.add_argument('--num_images', default=10, type=int,
                    help="The number of images to generate")
parser.add_argument('--ratio_of_2d_to_3d', default=100, type=int)
parser.add_argument('--turn_full_img_axis_on', action='store_true', default=True)

# Json Info
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
                    help="String to store in the \"date\" field of the generated JSON file;"
                         "defaults to today's date")
parser.add_argument('--version', default='1.0',
                    help="String to store in the \"version\" field of the generated JSON file")

# Plane Info
parser.add_argument('--plane_area_rate', default=[[0.10, 0.60], [0.10, 0.30], [0.10, 0.20], [0.05, 0.15], [0.03, 0.12]],
                    type=list)
parser.add_argument('--min_angle', default=25.0, type=float, help='The angle of the smallest angle of the triangle.')
parser.add_argument('--min_rect_asp_rate', default=0.3, type=float, help='The minimum aspect ratio of rectangle.')
parser.add_argument('--max_plane_retries', default=30000, type=int,
                    help='Maximum number of attempts to generate all plane geometries.')
parser.add_argument('--min_dist_plane', default=20.0, type=float,
                    help='The minimum distance between any two plane graphics in the image.')
parser.add_argument('--margin_x_plane', default=30.0, type=float,
                    help='The minimum distance between the centers of any two plane graphics along the x-axis.')
parser.add_argument('--margin_y_plane', default=30.0, type=float,
                    help='The minimum distance between the centers of any two plane graphics along the y-axis.')
parser.add_argument('--dist_to_axis', default=2.0, type=float,
                    help='The minimum distance between the coordinates of a plane point and the coordinate axis.'
                         'Prevent the pixel translation when the plane intersects with the coordinate axis.')

# Object Info
parser.add_argument('--max_obj_retries', default=20000, type=int,
                    help='Maximum number of attempts to generate all objects geometries.')
# parser.add_argument('--r_obj', default=5.0, type=float,
#                     help='The radius of the circumcircle of the small object geometry.')
# parser.add_argument('--zoom', default=2.5, type=float,
#                     help='The center of the circumcircle of the large-size object graphics is the zoom multiple '
#                          'of the small-size.')
parser.add_argument('--min_num_obj', default=5, type=int,
                    help='Minimum number of object geometry in plane geometry')
parser.add_argument('--max_num_obj', default=10, type=int,
                    help='Maximum number of object geometry in plane geometry')
parser.add_argument('--min_num_obj_ng', default=8, type=int,
                    help='Minimum number of geometric objects in the non-geometric figure(white space).')
parser.add_argument('--max_num_obj_ng', default=12, type=int,
                    help='Maximum number of geometric objects in a plane non-geometric figure(white space).')
parser.add_argument('--min_dist_obj', default=10.0, type=float,
                    help='The minimum distance between any two object graphics.')
parser.add_argument('--margin_x_obj_rate', default=0.25, type=float,
                    help='The overlapping length of the projections of any two object graphics in the x-axis '
                         'direction does not exceed the margin_x_obj_rate ratio of the shortest length from '
                         'the center of the object graphics to their own overlapping vertex.')
parser.add_argument('--margin_y_obj_rate', default=0.35, type=float,
                    help='The overlapping length of the projections of any two object graphics in the y-axis '
                         'direction does not exceed the margin_x_obj_rate ratio of the shortest length from '
                         'the center of the object graphics to their own overlapping vertex.')
parser.add_argument('--max_obj_colors', default=3, type=int)
parser.add_argument('--min_obj_colors', default=2, type=int)
parser.add_argument('--max_obj_materials', default=3, type=int)
parser.add_argument('--min_obj_materials', default=2, type=int)
parser.add_argument('--max_obj_3d_shapes', default=3, type=int)
parser.add_argument('--min_obj_3d_shapes', default=2, type=int)
parser.add_argument('--obj_line_width', default=0.2, type=float)
parser.add_argument('--obj_has_all_2d_shapes', action='store_true', default=False,
                    help='If True, the generated images must have circle, triangle and square.'
                         'If False, each generated image has at least one random shape among circle, '
                         'triangle and square')


def main():
    # This will give ground-truth information about the scene and its objects and planes
    scene = {'image_index': last_idx,
             'planes': [],
             'plane_relations': {
                 'left': [],
                 'right': [],
                 'up': [],
                 'down': []
             },
             'objects': [],
             'obj_relations': {
                 'left': [],
                 'right': [],
                 'up': [],
                 'down': []
             }}

    total_obj_materials = [*properties["obj_materials"]]
    assert len(total_obj_materials) >= args.max_obj_materials
    assert args.min_obj_materials >= 1 and not args.max_obj_materials < args.min_obj_materials
    obj_materials = random.sample(total_obj_materials, random.randint(args.min_obj_materials, args.max_obj_materials))

    total_obj_colors = [*properties["obj_colors"]]
    assert len(total_obj_colors) >= args.max_obj_colors
    assert args.min_obj_colors >= 1 and not args.max_obj_colors < args.min_obj_colors
    obj_colors = random.sample(total_obj_colors, random.randint(args.min_obj_colors, args.max_obj_colors))

    obj_3d_shapes_total = []
    obj_2d_shapes_total = []
    for s in list(properties["obj_shapes"]):
        for v in list(properties["obj_shapes"][s]):
            obj_3d_shapes_total.extend([v])
            obj_2d_shapes_total.extend([s])

    assert len(obj_3d_shapes_total) >= args.max_obj_3d_shapes
    assert args.min_obj_3d_shapes >= 1 and not args.max_obj_3d_shapes < args.min_obj_3d_shapes
    num_shapes = random.randint(args.min_obj_3d_shapes, args.max_obj_3d_shapes)
    if args.obj_has_all_2d_shapes:
        a = [*properties["obj_shapes"]]
        assert args.min_obj_3d_shapes >= len(a)
        shape_lists = []
        for s in a:
            shape_lists.append([*properties["obj_shapes"][s]])
        obj_3d_shapes = []
        obj_2d_shapes = []
        for s in range(len(shape_lists)):
            rdm = random.sample(shape_lists[s], 1)
            obj_3d_shapes.extend(rdm)
            obj_2d_shapes.extend([a[s]])
            s_id = obj_3d_shapes_total.index(rdm[0])
            obj_3d_shapes_total.pop(s_id)
            obj_2d_shapes_total.pop(s_id)
        if num_shapes > len(a):
            s_3, s_2 = zip(*random.sample(list(zip(obj_3d_shapes_total, obj_2d_shapes_total)),
                                          num_shapes - len(a)))
            obj_3d_shapes.extend(list(s_3))
            obj_2d_shapes.extend(list(s_2))
    else:
        obj_3d_shapes, obj_2d_shapes = zip(*random.sample(list(zip(obj_3d_shapes_total, obj_2d_shapes_total)),
                                                          num_shapes))
        obj_3d_shapes, obj_2d_shapes = list(obj_3d_shapes), list(obj_2d_shapes)

    # Generate plane graphics
    shapes = utils.gen_planes(args, properties)

    # Generate small geometric figures (objects) in big geometry (plane).
    obj_index, planes_objs, shapes = utils.gen_objs_in_plane(args,
                                                             properties,
                                                             obj_2d_shapes,
                                                             obj_3d_shapes,
                                                             shapes,
                                                             obj_materials,
                                                             obj_colors
                                                             )

    # Generate small graphics (objects) in blank areas (non-geometric plane).
    bg_objs = utils.gen_objs_in_ng_plane(
        args,
        properties,
        shapes,
        obj_2d_shapes,
        obj_3d_shapes,
        obj_index,
        obj_materials,
        obj_colors,
        planes_objs
    )

    # Generating image
    img_width = properties["plane_size"]["width"] * args.ratio_of_2d_to_3d
    img_length = properties["plane_size"]["length"] * args.ratio_of_2d_to_3d
    plt.figure()
    ax = plt.axes()
    plt.axis('scaled')
    plt.xlim((0, img_length))
    plt.ylim((0, img_width))

    #  Display planes, used for 3d material grayscale image
    for shape in shapes:
        material = random.choice([*properties["plane_materials_grayscale"]])
        shape.material = material
        shape.fc = tuple(properties["plane_materials_grayscale"][material])
        p = shape.to_patch()
        ax.add_patch(p)
    print('save 2D scene -- used for the 3D floor materials')
    plt.gca().set_axis_off()
    plt.savefig(grayscale_img2d_path, bbox_inches='tight', pad_inches=0)

    # Display the image only contains planes
    ax.patches = []
    for shape in shapes:
        shape.fc = 'none'
        p = shape.to_patch()
        ax.add_patch(p)
        color = random.choice(properties["plane_colors"])
        shape.color = color
        # Add material texture on plane
        img = plt.imread(os.path.join(args.plane_mat_col_dir, f'{color}_{shape.material}.jpeg'))
        im = ax.imshow(img, extent=[0, img_length, 0, img_width])
        im.set_clip_path(p)
        print(shape, shape.color, shape.material)
    print('save 2D scene')
    plt.savefig(img2d_path, bbox_inches='tight', pad_inches=0)

    # Display the image contains planes and objects
    for shape in shapes:
        for obj in shape.objs:
            if args.obj_line_width is not None:
                obj.ec = 'white'
                obj.lw = args.obj_line_width
            p = obj.to_patch()
            ax.add_patch(p)

    for obj in bg_objs:
        p = obj.to_patch()
        ax.add_patch(p)

    print('save full 2D scene')
    if args.turn_full_img_axis_on:
        plt.gca().set_axis_on()
        plt.savefig(full_img2d_path)
    else:
        plt.savefig(full_img2d_path, bbox_inches='tight', pad_inches=0)
    plt.close('all')

    # Output json file
    total_objs = []
    directions = ['left', 'right', 'up', 'down']

    # Information about planes
    for shape in shapes:
        total_objs.extend(shape.objs)
        scene['planes'].append({
            'shape': shape.shape,
            'coordinate_points': shape.coordinate_points(),
            'center_coordinate': shape.center(),
            'material': shape.material,
            'color': shape.color,
            'area_ratio': shape.area_ratio,
            'internal_objs': [obj.index for obj in shape.objs]
        })
        shape.get_adj(shapes)
        for d in directions:
            scene['plane_relations'][d].append(shape.adj[d])

    # Information about non-geometric area
    scene['planes'].append({
        'shape': 'non-geometric',
        'coordinate_points': '',
        'center_coordinate': '',
        'material': '',
        'color': 'white',
        'area_ratio': '',
        'internal_objs': [obj.index for obj in bg_objs]
    })
    total_objs.extend(bg_objs)

    # Information about objects
    for obj in total_objs:
        scene['objects'].append({
            'coordinate_points': obj.coordinate_points(),
            'center_coordinate': obj.center(),
            'rotation': obj.rotation,
            'size': obj.size,
            'color': obj.color,
            'material': obj.material,
            'shape': obj.shape,
            'shape_3d': obj.shape_3d,
        })
        obj.get_adj(total_objs)
        for d in directions:
            scene['obj_relations'][d].append(obj.adj[d])

    json_dic['scenes'].append(scene)
    with open(json_path, 'w', encoding='utf-8') as file:
        # json.dump(json_dic, file, indent=1)
        json.dump(json_dic, file)
    print(f'Save picture and json file {last_idx}')


if __name__ == '__main__':

    args = parser.parse_args()

    # check and create output dir
    output_path = os.path.join(args.output_dir, args.split)
    img_path = os.path.join(output_path, args.img_dir)
    full_img_path = os.path.join(output_path, args.full_img_dir)
    img_mat_path = os.path.join(output_path, args.img_mat_dir)
    json_prefix = '%s_%s.json' % (args.json_prefix, args.split)
    json_path = os.path.join(output_path, json_prefix)

    num_digits = 6
    img2d_prefix = '%s_%s_' % (args.img_prefix, args.split)
    img2d_template = '%s%%0%dd.png' % (img2d_prefix, num_digits)
    img2d_template = os.path.join(img_path, img2d_template)
    grayscale_prefix = '%s_%s_' % (args.img_mat_prefix, args.split)
    grayscale_img_template = '%s%%0%dd.png' % (grayscale_prefix, num_digits)
    grayscale_img_template = os.path.join(img_mat_path, grayscale_img_template)
    full2d_prefix = '%s_%s_' % (args.full_img_prefix, args.split)
    full2d_img_template = '%s%%0%dd.png' % (full2d_prefix, num_digits)
    full2d_img_template = os.path.join(full_img_path, full2d_img_template)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(img_mat_path):
        os.mkdir(img_mat_path)
    if not os.path.exists(full_img_path):
        os.mkdir(full_img_path)
    if not os.path.exists(json_path):
        with open(json_path, 'w', encoding='utf-8') as _:
            json_dic = {
                'info': {
                    'date': args.date,
                    'version': args.version,
                    'split': args.split
                },
                'scenes': []
            }
    else:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_dic = json.load(f)
        if args.start_idx is not None:
            if json_dic['scenes'][-1]['image_index'] >= args.start_idx:
                assert json_dic['scenes'][
                           args.start_idx - (json_dic['scenes'][-1]['image_index'] - len(json_dic['scenes']) + 1)][
                           'image_index'] == args.start_idx, \
                    "The index of the json information is conflicts with the start index of the image to be generated."
                print(" * The image and json information starting with the 'start_idx' already exist in the directory.")
                print(" * Do you want to overwrite this information and continue to generate? (y/n)")
                in_content = input("Input：")
                if in_content == "y":
                    json_dic['scenes'] = json_dic['scenes'][:args.start_idx - (
                                json_dic['scenes'][-1]['image_index'] - len(json_dic['scenes']) + 1)]
                    for filename in os.listdir(img_path):
                        index = filename.split('_')[-1]
                        index = int(index.split('.')[0])
                        if index >= args.start_idx:
                            os.remove(img2d_template % index)
                            os.remove(full2d_img_template % index)
                            os.remove(grayscale_img_template % index)
                else:
                    sys.exit()
            else:
                assert json_dic['scenes'][-1]['image_index'] == args.start_idx - 1, \
                    "The 'start_idx' is not close to the previous one. Please confirm whether the 'start_idx' is correct."

    assert os.path.exists(args.properties_json)
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)

    # Last index number of images
    # indices = []
    # for filename in os.listdir('2D_scene/images'):
    #     index = filename.split('_')[-1]
    #     index = int(index.split('.')[0])
    #     indices.append(index)
    # if len(indices) > 0:
    #     last_idx = max(indices) + 1
    # else:
    #     last_idx = 0

    if args.start_idx is not None:
        last_idx = args.start_idx
    else:
        last_idx = 0
        if os.listdir(img_path):
            # This directory is not empty
            for filename in os.listdir(img_path):
                index = filename.split('_')[-1]
                index = int(index.split('.')[0])
                last_idx = index + 1 if index >= last_idx else last_idx

    for i in range(args.num_images):
        img2d_path = img2d_template % last_idx
        grayscale_img2d_path = grayscale_img_template % last_idx
        full_img2d_path = full2d_img_template % last_idx

        main()
        last_idx += 1
        print()
