import bpy, bpy_extras
import os
import math
import random
import sys
from mathutils import Vector
import tempfile
from collections import Counter

"""
Some utility functions for interacting with Blender
"""


def extract_args(input_argv=None):
    """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if '--' in input_argv:
        idx = input_argv.index('--')
        output_argv = input_argv[(idx + 1):]
    return output_argv


def parse_args(parser, argv=None):
    return parser.parse_args(extract_args(argv))


def rand(L):
    return 2.0 * L * (random.random() - 0.5)


def set_camera(args, scene_3d, cam_pos=None):
    if cam_pos is not None:
        index = int(cam_pos)
    else:
        index = random.randint(0, len(args.camera_position) - 1)

    # Add random jitter to camera position
    camera_position = args.camera_position[index]
    camera_rotation = args.camera_rotation[index]
    for i in range(3):
        bpy.data.objects['Camera'].location[i] = camera_position[i]
        bpy.data.objects['Camera'].rotation_euler[i] = math.radians(camera_rotation[i])
    if args.camera_loc_jitter > 0 and args.camera_rtt_jitter > 0:
        for i in range(3):
            bpy.data.objects['Camera'].location[i] += rand(args.camera_loc_jitter)
            bpy.data.objects['Camera'].rotation_euler[i] += math.radians(rand(args.camera_rtt_jitter))
    scene_3d['blender_info']['camera_lct'] = tuple(bpy.data.objects['Camera'].location)
    scene_3d['blender_info']['camera_rtt'] = [math.degrees(bpy.data.objects['Camera'].rotation_euler[i]) for i in
                                              range(3)]
    print(f'camera position index {index}')
    return index


def set_light(args):
    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
            bpy.data.objects['Lamp_Key'].rotation_euler[i] += math.radians(rand(args.light_rtt_jitter))
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
            bpy.data.objects['Lamp_Back'].rotation_euler[i] += math.radians(rand(args.light_rtt_jitter))
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)
            bpy.data.objects['Lamp_Fill'].rotation_euler[i] += math.radians(rand(args.light_rtt_jitter))
    if args.light_power_jitter > 0:
        bpy.data.lights['Lamp_Key'].energy += rand(args.light_power_jitter)
        bpy.data.lights['Lamp_Back'].energy += rand(args.light_power_jitter)
        bpy.data.lights['Lamp_Fill'].energy += rand(args.light_power_jitter)


def set_render(output_image, args):
    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size

    if args.use_gpu:
        # Blender changed the API for enabling CUDA at some point
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'OPTIX'
        bpy.context.scene.cycles.device = 'GPU'

    # Some CYCLES-specific stuff
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.view_layer.cycles.denoising_store_passes = True
    # render_args.film_transparent = True
    # bpy.context.scene.cycles.film_transparent_glass = True
    bpy.context.scene.cycles.min_transparent_bounces = args.render_transparent_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_transparent_max_bounces
    bpy.context.scene.cycles.max_bounces = args.render_total_max_bounces
    bpy.context.scene.cycles.transmission_bounces = args.render_transmission_bounces
    bpy.context.scene.cycles.blur_glossy = 1.0


def load_materials(material_dir):
    """
  Load materials from a directory. We assume that the directory contains .blend
  files with one material each. The file X.blend has a single NodeTree item named
  X; this NodeTree item must have a "Color" input that accepts an RGBA value.
  """
    for fn in os.listdir(material_dir):
        if not fn.endswith('.blend'):
            continue
        name = os.path.splitext(fn)[0]
        filepath = os.path.join(material_dir, fn, 'NodeTree', name)
        bpy.ops.wm.append(filename=filepath)

    # if "Material_0" in bpy.data.materials:
    #     # mat = bpy.data.materials['Material']
    #     # mat.name = 'Material.000'
    #
    # for i in bpy.data.materials:
    #     i.use_fake_user = True


def add_material(name, **properties):
    """
    Create a new material and assign it to the active object. "name" should be the
    name of a material that has been previously loaded using load_materials.
    """
    # Figure out how many materials are already in the scene
    mat_count = len(bpy.data.materials)

    # Create a new material; it is not attached to anything and
    # it will be called "Material"
    bpy.ops.material.new()

    # Get a reference to the material we just created and rename it;
    # then the next time we make a new material it will still be called
    # "Material" and we will still be able to look it up by name
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % mat_count

    # Delete default node
    mat.node_tree.nodes.remove(mat.node_tree.nodes['Principled BSDF'])

    # Attach the new material to the active object
    # Make sure it doesn't already have materials
    obj = bpy.context.active_object
    assert len(obj.data.materials) == 0
    obj.data.materials.append(mat)

    # Find the output node of the new material
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'Material Output':
            output_node = n
            break

    # Add a new GroupNode to the node tree of the active material,
    # and copy the node tree from the preloaded node group to the
    # new group node. This copying seems to happen by-value, so
    # we can create multiple materials of the same type without them
    # clobbering each other
    group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
    group_node.node_tree = bpy.data.node_groups[name]

    # Find and set the "Color" input of the new group node
    for inp in group_node.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    # Wire the output of the new group node to the input of
    # the MaterialOutput node
    mat.node_tree.links.new(
        group_node.outputs['Shader'],
        output_node.inputs['Surface'],
    )


def create_plane(length, width, img_path, grayscale_img_path):
    # Create a plane on the ground
    # bpy.ops.mesh.primitive_plane_add(size=1, location=[length / 2, width / 2, 0], scale=[1, 1, 1])
    bpy.context.view_layer.objects.active = bpy.data.objects['Plane']
    plane = bpy.context.object

    # Change the size of the plane to match the size ratio of 2D image
    plane.scale[0] = length / 2
    plane.scale[1] = width / 2
    plane.location[0] = length / 2
    plane.location[1] = width / 2

    # Check how many materials are already in the scene
    mat_count = len(bpy.data.materials)

    # Create a new material for plane; it is not attached to anything and
    # it will be called "Material"
    bpy.ops.material.new()

    # Get a reference to the material we just created and rename it;
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % mat_count

    # Creating a node for a shader network
    mat_nodes = mat.node_tree.nodes
    texture = mat_nodes.new('ShaderNodeTexImage')

    # Load 2D Image
    bpy.data.images.load(img_path, check_existing=True)
    img_name = os.path.basename(os.path.normpath(img_path))

    # Setting an image in the image texture node
    texture.image = bpy.data.images[img_name]

    # Assign texture to material's displacement
    mat.node_tree.links.new(
        mat_nodes['Principled BSDF'].inputs["Base Color"],
        texture.outputs['Color'])

    # Creating another node for a shader network
    grayscale_texture = mat_nodes.new('ShaderNodeTexImage')

    # Load 2D material grayscale Image
    bpy.data.images.load(grayscale_img_path, check_existing=True)
    grayscale_img_name = os.path.basename(os.path.normpath(grayscale_img_path))

    # Setting grayscale image in the image texture node
    grayscale_texture.image = bpy.data.images[grayscale_img_name]

    # Assign grayscale texture to material's displacement
    mat.node_tree.links.new(
        mat_nodes['Principled BSDF'].inputs["Roughness"],
        grayscale_texture.outputs['Color'])

    # Attach the new material to the plane
    # Make sure it doesn't already have materials
    assert len(plane.data.materials) == 0
    plane.data.materials.append(mat)


def get_camera_coords(cam, pos):
    """
  For a specified point, get both the 3D coordinates and 2D pixel-space
  coordinates of the point from the perspective of the camera.

  Inputs:
  - cam: Camera object
  - pos: Vector giving 3D world-space position

  Returns a tuple of:
  - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
    in the range [-1, 1]
  """
    scene = bpy.context.scene
    x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
    scale = scene.render.resolution_percentage / 100.0
    w = round(scale * scene.render.resolution_x)
    h = round(scale * scene.render.resolution_y)
    px = round(x * w)
    py = round(h - y * h)
    if px < 0:
        if py < 0:
            return (0.0, 0.0, z)
        elif py > h:
            return (0.0, h, z)
        else:
            return (0.0, py, z)
    elif px > w:
        if py < 0:
            return (w, 0.0, z)
        elif py > h:
            return (w, h, z)
        else:
            return (w, py, z)
    else:
        return (px, py, z)


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :param me_ob: Untransformed Mesh.
    :type me: :class:'bpy.types.Mesh'
    :type obj: :class:'bpy.types.Object'
    :param cam_ob: Camera object.
    :param scene: Scene to use for frame size.
    :type scene: :class:'bpy.types.Scene'

    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:Box
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = me_ob.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            if z <= 0.0:
                """ Vertex is behind the camera; ignore it. """
                continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    mesh_eval.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    # Sanity check, image is not in view if both bounding points exist on the same side
    if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
        return (0, 0, 0, 0)

    return (
        round(min_x * dim_x),  # X
        round(dim_y - max_y * dim_y),  # Y
        round((max_x - min_x) * dim_x),  # Width
        round((max_y - min_y) * dim_y)  # Height
    )


def plane_bounding_box(plane, camera, image_ration):
    if plane['shape'] == 'rectangle':
        coord1 = plane['coordinate_points'][0].copy()
        coord2 = plane['coordinate_points'][1].copy()
        coord3 = plane['coordinate_points'][2].copy()
        coord4 = plane['coordinate_points'][3].copy()
        coord1.append(0.0)
        coord1 = Vector([x / image_ration for x in coord1])
        coord1 = get_camera_coords(camera, coord1)
        coord2.append(0.0)
        coord2 = Vector([x / image_ration for x in coord2])
        coord2 = get_camera_coords(camera, coord2)
        coord3.append(0.0)
        coord3 = Vector([x / image_ration for x in coord3])
        coord3 = get_camera_coords(camera, coord3)
        coord4.append(0.0)
        coord4 = Vector([x / image_ration for x in coord4])
        coord4 = get_camera_coords(camera, coord4)
        left = min([coord1[0], coord2[0], coord3[0], coord4[0]])
        top = min([coord1[1], coord2[1], coord3[1], coord4[1]])
        width = max([coord1[0], coord2[0], coord3[0], coord4[0]]) - left
        height = max([coord1[1], coord2[1], coord3[1], coord4[1]]) - top
    elif plane['shape'] == 'non-geometric':
        coord1 = [-50, -50]
        coord2 = [-50, 450]
        coord3 = [450, 450]
        coord4 = [450, -50]
        coord1.append(0.0)
        coord1 = Vector([x / image_ration for x in coord1])
        coord1 = get_camera_coords(camera, coord1)
        coord2.append(0.0)
        coord2 = Vector([x / image_ration for x in coord2])
        coord2 = get_camera_coords(camera, coord2)
        coord3.append(0.0)
        coord3 = Vector([x / image_ration for x in coord3])
        coord3 = get_camera_coords(camera, coord3)
        coord4.append(0.0)
        coord4 = Vector([x / image_ration for x in coord4])
        coord4 = get_camera_coords(camera, coord4)
        left = min([coord1[0], coord2[0], coord3[0], coord4[0]])
        top = min([coord1[1], coord2[1], coord3[1], coord4[1]])
        width = max([coord1[0], coord2[0], coord3[0], coord4[0]]) - left
        height = max([coord1[1], coord2[1], coord3[1], coord4[1]]) - top
    elif plane['shape'] == 'triangle':
        coord1 = plane['coordinate_points'][0].copy()
        coord2 = plane['coordinate_points'][1].copy()
        coord3 = plane['coordinate_points'][2].copy()
        coord1.append(0.0)
        coord1 = Vector([x / image_ration for x in coord1])
        coord1 = get_camera_coords(camera, coord1)
        coord2.append(0.0)
        coord2 = Vector([x / image_ration for x in coord2])
        coord2 = get_camera_coords(camera, coord2)
        coord3.append(0.0)
        coord3 = Vector([x / image_ration for x in coord3])
        coord3 = get_camera_coords(camera, coord3)
        left = min([coord1[0], coord2[0], coord3[0]])
        top = min([coord1[1], coord2[1], coord3[1]])
        width = max([coord1[0], coord2[0], coord3[0]]) - left
        height = max([coord1[1], coord2[1], coord3[1]]) - top
    else:
        center = plane['coordinate_points'][0].copy()
        r = plane['coordinate_points'][1]
        center.append(0.0)
        coord1 = (center[0] - r, center[1], center[2])
        coord1 = Vector([x / image_ration for x in coord1])
        coord1 = get_camera_coords(camera, coord1)
        coord2 = (center[0] + r, center[1], center[2])
        coord2 = Vector([x / image_ration for x in coord2])
        coord2 = get_camera_coords(camera, coord2)
        coord3 = (center[0], center[1] - r, center[2])
        coord3 = Vector([x / image_ration for x in coord3])
        coord3 = get_camera_coords(camera, coord3)
        coord4 = (center[0], center[1] + r, center[2])
        coord4 = Vector([x / image_ration for x in coord4])
        coord4 = get_camera_coords(camera, coord4)
        left = coord1[0]
        top = coord4[1]
        width = coord2[0] - left
        height = coord3[1] - top
    return (left, top, width, height)


def add_objects(scene_info, properties, camera, args):
    img_ratio = args.ratio_of_2d_to_3d
    object_dir = args.shape_dir
    obj_json = []
    # Load the property file
    color_name_to_rgba = {}
    blender_objects = []
    for name, rgb in properties['obj_colors'].items():
        rgba = [float(c) / 255.0 for c in rgb] + [1.0]
        color_name_to_rgba[name] = rgba
    material_mapping = properties['obj_materials']
    obj_shapes = list(properties["obj_shapes"].values())
    object_mapping = {}
    for i in range(len(obj_shapes)):
        object_mapping.update(obj_shapes[i])
    size_mapping = properties['sizes']

    obj_info = scene_info["objects"]
    for i in range(len(obj_info)):
        size = obj_info[i]["size"]
        # 320 x 240 2D image has been resized to img_ratio
        r = size_mapping[size] / img_ratio

        shape = obj_info[i]["shape_3d"]
        shape_blend = object_mapping[shape]

        color = obj_info[i]["color"]
        rgba = color_name_to_rgba[color]

        material = obj_info[i]["material"]
        material_blend = material_mapping[material]
        theta = obj_info[i]["rotation"]

        # First figure out how many of this object are already in the scene so we can
        # give the new object a unique name
        count = 0
        for obj in bpy.data.objects:
            if obj.name.startswith(shape_blend):
                count += 1

        filename = os.path.join(object_dir, '%s.blend' % shape_blend, 'Object', shape_blend)
        bpy.ops.wm.append(filename=filename)

        # Give it a new name to avoid conflicts
        new_name = '%s_%d' % (shape_blend, count)
        bpy.data.objects[shape_blend].name = new_name

        # Set the new object as active, then rotate, scale, and translate it
        x, y = obj_info[i]["center_coordinate"]
        # bpy.context.scene.objects.active = bpy.data.objects[new_name]
        bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
        # rotation_z = round(math.degrees(bpy.context.object.rotation_euler[2]) ,0) + theta
        # bpy.context.object.rotation_euler.rotate_axis("Z", math.radians(rotation_z))
        obj = bpy.context.object
        obj.rotation_euler.rotate_axis("Z", math.radians(360.0 - theta))

        if shape_blend == 'Cube':
            r *= math.sqrt(2) / 2
        bpy.ops.transform.resize(value=[r, r, r])
        bpy.ops.transform.translate(value=[x / img_ratio, y / img_ratio, r])

        # Attach color and material
        if isinstance(material_blend, list):
            mat_name = random.choice(material_blend)
        else:
            mat_name = material_blend
        add_material(mat_name, Color=rgba)

        # Record data about the object in the scene data structure
        pixel_coords = get_camera_coords(camera, obj.location)

        # Get 2D pixel coordinates for all 8 points in the bounding box
        bound_box = camera_view_bounds_2d(bpy.context.scene, camera, obj)

        obj_json.append({
            'rotation': obj_info[i]['rotation'],
            'size': obj_info[i]['size'],
            'color': obj_info[i]['color'],
            'material': obj_info[i]['material'],
            'shape': obj_info[i]['shape_3d'],
            'pixel_coords': pixel_coords,
            '3d_coords': tuple(obj.location),
            'bounding_box': {
                'left': bound_box[0],
                'top': bound_box[1],
                'width': bound_box[2],
                'height': bound_box[3]
            }
        })
        blender_objects.append(obj)

    return obj_json, blender_objects


def compute_spatial_relationships(scene_3d, scene_2d, img_ratio, eps_obj=0.025, eps_plane=0.15):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    for name, direction_vec in scene_3d['blender_info']['directions'].items():
        if name == 'above' or name == 'below': continue
        scene_3d['obj_spatial_relations'][name] = []
        for i, obj1 in enumerate(scene_3d['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_3d['objects']):
                if obj1 == obj2: continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps_obj:
                    related.add(j)
            scene_3d['obj_spatial_relations'][name].append(sorted(list(related)))

        scene_3d['plane_spatial_relations'][name] = []
        for i, p1 in enumerate(scene_2d['planes']):
            if type(p1['center_coordinate']) not in [list, tuple]: continue
            cc1 = p1['center_coordinate'].copy()
            cc1.append(0.0)
            p_coords1 = [x / img_ratio for x in cc1]
            related = set()
            for j, p2 in enumerate(scene_2d['planes']):
                if type(p2['center_coordinate']) not in [list, tuple]: continue
                if p1 == p2: continue
                cc2 = p2['center_coordinate'].copy()
                cc2.append(0.0)
                p_coords2 = [x / img_ratio for x in cc2]
                diff = [p_coords2[k] - p_coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps_plane:
                    related.add(j)
            scene_3d['plane_spatial_relations'][name].append(sorted(list(related)))


# def compute_pixel_relationships(scene_3d, eps_obj=2, eps_plane=2):
#     for i, obj1 in enumerate(scene_3d['objects']):
#         coords1 = []
#         coords1.append(round(obj1['bounding_box']['left'] + obj1['bounding_box']['width'] / 2))
#         coords1.append(round(obj1['bounding_box']['top'] + obj1['bounding_box']['height'] / 2))
#         related_behind = set()
#         related_front = set()
#         related_left = set()
#         related_right = set()
#         for j, obj2 in enumerate(scene_3d['objects']):
#             if obj1 == obj2: continue
#             coords2 = []
#             coords2.append(round(obj2['bounding_box']['left'] + obj2['bounding_box']['width'] / 2))
#             coords2.append(round(obj2['bounding_box']['top'] + obj2['bounding_box']['height'] / 2))
#             if abs(coords1[0] - coords2[0]) >= eps_obj:
#                 if coords2[0] > coords1[0]:
#                     related_right.add(j)
#                 else:
#                     related_left.add(j)
#             if abs(coords1[1] - coords2[1]) >= eps_obj:
#                 if coords2[1] > coords1[1]:
#                     related_front.add(j)
#                 else:
#                     related_behind.add(j)
#         scene_3d['obj_pixel_relations']['behind'].append(sorted(list(related_behind)))
#         scene_3d['obj_pixel_relations']['front'].append(sorted(list(related_front)))
#         scene_3d['obj_pixel_relations']['left'].append(sorted(list(related_left)))
#         scene_3d['obj_pixel_relations']['right'].append(sorted(list(related_right)))
#
#     for i, p1 in enumerate(scene_3d['planes']):
#         if type(p1['bounding_box']) is not dict: continue
#         cc1 = []
#         cc1.append(round(p1['bounding_box']['left'] + p1['bounding_box']['width'] / 2))
#         cc1.append(round(p1['bounding_box']['top'] + p1['bounding_box']['height'] / 2))
#         related_behind = set()
#         related_front = set()
#         related_left = set()
#         related_right = set()
#         for j, p2 in enumerate(scene_3d['planes']):
#             if type(p2['bounding_box']) is not dict: continue
#             if p1 == p2: continue
#             cc2 = []
#             cc2.append(round(p2['bounding_box']['left'] + p2['bounding_box']['width'] / 2))
#             cc2.append(round(p2['bounding_box']['top'] + p2['bounding_box']['height'] / 2))
#             if abs(cc1[0] - cc2[0]) >= eps_plane:
#                 if cc2[0] > cc1[0]:
#                     related_right.add(j)
#                 else:
#                     related_left.add(j)
#             if abs(cc1[1] - cc2[1]) >= eps_plane:
#                 if cc2[1] > cc1[1]:
#                     related_front.add(j)
#                 else:
#                     related_behind.add(j)
#         scene_3d['plane_pixel_relations']['behind'].append(sorted(list(related_behind)))
#         scene_3d['plane_pixel_relations']['front'].append(sorted(list(related_front)))
#         scene_3d['plane_pixel_relations']['left'].append(sorted(list(related_left)))
#         scene_3d['plane_pixel_relations']['right'].append(sorted(list(related_right)))


def compute_pixel_relationships(scene_3d, eps_obj=2, eps_plane=4):
    for i, obj1 in enumerate(scene_3d['objects']):
        coords1 = obj1['pixel_coords']
        related_behind = set()
        related_front = set()
        related_left = set()
        related_right = set()
        for j, obj2 in enumerate(scene_3d['objects']):
            if obj1 == obj2: continue
            coords2 = obj2['pixel_coords']
            if abs(coords1[0] - coords2[0]) >= eps_obj:
                if coords2[0] > coords1[0]:
                    related_right.add(j)
                else:
                    related_left.add(j)
            if abs(coords1[1] - coords2[1]) >= eps_obj:
                if coords2[1] > coords1[1]:
                    related_front.add(j)
                else:
                    related_behind.add(j)
        scene_3d['obj_pixel_relations']['behind'].append(sorted(list(related_behind)))
        scene_3d['obj_pixel_relations']['front'].append(sorted(list(related_front)))
        scene_3d['obj_pixel_relations']['left'].append(sorted(list(related_left)))
        scene_3d['obj_pixel_relations']['right'].append(sorted(list(related_right)))

    for i, p1 in enumerate(scene_3d['planes']):
        if type(p1['center_pixel_coords']) not in [list, tuple]: continue
        cc1 = p1['center_pixel_coords']
        related_behind = set()
        related_front = set()
        related_left = set()
        related_right = set()
        for j, p2 in enumerate(scene_3d['planes']):
            if type(p2['center_pixel_coords']) not in [list, tuple]: continue
            if p1 == p2: continue
            cc2 = p2['center_pixel_coords']
            if abs(cc1[0] - cc2[0]) >= eps_plane:
                if cc2[0] > cc1[0]:
                    related_right.add(j)
                else:
                    related_left.add(j)
            if abs(cc1[1] - cc2[1]) >= eps_plane:
                if cc2[1] > cc1[1]:
                    related_front.add(j)
                else:
                    related_behind.add(j)
        scene_3d['plane_pixel_relations']['behind'].append(sorted(list(related_behind)))
        scene_3d['plane_pixel_relations']['front'].append(sorted(list(related_front)))
        scene_3d['plane_pixel_relations']['left'].append(sorted(list(related_left)))
        scene_3d['plane_pixel_relations']['right'].append(sorted(list(related_right)))


def check_visibility(blender_objects, min_pixels_per_object, use_gpu):
    """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
    f, path = tempfile.mkstemp(suffix='.png')
    if use_gpu:
        object_colors = render_shadeless_gpu(blender_objects, path=path)
    else:
        object_colors = render_shadeless_cpu(blender_objects, path=path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    color_count = Counter((p[i], p[i + 1], p[i + 2], p[i + 3])
                          for i in range(0, len(p), 4))
    bpy.data.images.remove(img)
    os.remove(path)
    assert len(object_colors) == len(blender_objects), f'The number of objects ({len(blender_objects)}) ' \
                                                       f'is not equal to the number of object' \
                                                       f' colors ({len(object_colors) - 1}).'
    if len(color_count) != len(object_colors) + 1:
        print(f'The color count ({len(color_count) - 1}) of shadeless image is not equal to the object colors '
              f'({len(object_colors)}), which means some objects has been totally occluded. '
              f'Will rerender this image with different camera postion. ')
        return False
    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            print(f'Here is one object whose color count ({count}) is less than your min_pixels_per_object '
                  f'({min_pixels_per_object}) setting. Will rerender this image with different camera postion.')
            return False
    return True


def render_shadeless_gpu(blender_objects, path='flat.png'):
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_denoise_enabled = bpy.context.scene.node_tree.nodes['Denoise'].mute
    old_render_samples = bpy.context.scene.cycles.samples
    old_use_adaptive_sampling = bpy.context.scene.cycles.use_adaptive_sampling
    old_denoising_store_passes = bpy.context.view_layer.cycles.denoising_store_passes
    old_dither = bpy.context.scene.render.dither_intensity

    # Override some render settings to have flat shading
    render_args.filepath = path
    bpy.context.scene.node_tree.nodes['Denoise'].mute = True
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.cycles.use_adaptive_sampling = False
    bpy.context.view_layer.cycles.denoising_store_passes = False
    bpy.context.scene.render.dither_intensity = 0.0

    # Don't render lights and ground
    bpy.data.objects['Lamp_Key'].hide_render = True
    bpy.data.objects['Lamp_Fill'].hide_render = True
    bpy.data.objects['Lamp_Back'].hide_render = True
    bpy.data.objects['Sun'].hide_render = True
    bpy.data.objects['Plane'].hide_render = True
    bpy.data.objects['Ground'].hide_render = True

    # Add random shadeless materials to all objects
    object_colors = set()
    old_materials = []

    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        emmNode = mat.node_tree.nodes.new(type="ShaderNodeEmission")
        origNode = mat.node_tree.nodes["Principled BSDF"]
        OutputNode = mat.node_tree.nodes["Material Output"]
        mat.node_tree.nodes.remove(origNode)
        mat.node_tree.links.new(OutputNode.inputs['Surface'], emmNode.outputs['Emission'])
        while True:
            r, g, b = [round(random.random(), 1) for _ in range(3)]
            if (r, g, b, 1.0) not in object_colors: break
        object_colors.add((r, g, b, 1.0))
        emmNode.inputs['Color'].default_value = [r, g, b, 1.0]
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    bpy.data.objects['Lamp_Key'].hide_render = False
    bpy.data.objects['Lamp_Fill'].hide_render = False
    bpy.data.objects['Lamp_Back'].hide_render = False
    bpy.data.objects['Sun'].hide_render = False
    bpy.data.objects['Plane'].hide_render = False
    bpy.data.objects['Ground'].hide_render = False

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    bpy.context.scene.node_tree.nodes['Denoise'].mute = old_denoise_enabled
    bpy.context.scene.cycles.samples = old_render_samples
    bpy.context.scene.cycles.use_adaptive_sampling = old_use_adaptive_sampling
    bpy.context.view_layer.cycles.denoising_store_passes = old_denoising_store_passes
    bpy.context.scene.render.dither_intensity = old_dither

    return object_colors


def render_shadeless_cpu(blender_objects, path='flat.png'):
    """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = bpy.context.scene.display.render_aa
    old_view3DShading_light = bpy.data.scenes['Scene'].display.shading.light
    old_view3DShading_color_type = bpy.data.scenes['Scene'].display.shading.color_type
    # old_transparency = bpy.context.scene.render.film_transparent
    old_denoise_enabled = bpy.context.scene.node_tree.nodes['Denoise'].mute
    old_view_transform = bpy.context.scene.view_settings.view_transform
    old_dither = bpy.context.scene.render.dither_intensity

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'BLENDER_WORKBENCH'
    bpy.context.scene.display.render_aa = 'OFF'
    bpy.data.scenes['Scene'].display.shading.light = 'FLAT'
    bpy.data.scenes['Scene'].display.shading.color_type = 'TEXTURE'
    # bpy.context.scene.render.film_transparent = True
    bpy.context.scene.node_tree.nodes['Denoise'].mute = True
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.render.dither_intensity = 0.0

    # Don't render lights and ground
    bpy.data.objects['Lamp_Key'].hide_render = True
    bpy.data.objects['Lamp_Fill'].hide_render = True
    bpy.data.objects['Lamp_Back'].hide_render = True
    bpy.data.objects['Sun'].hide_render = True
    bpy.data.objects['Plane'].hide_render = True
    bpy.data.objects['Ground'].hide_render = True

    # Add random shadeless materials to all objects
    object_colors = set()
    old_materials = []

    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        while True:
            r, g, b = [round(random.random(), 1) for _ in range(3)]
            if (r, g, b, 1.0) not in object_colors: break
        object_colors.add((r, g, b, 1.0))
        mat.diffuse_color = [r, g, b, 1.0]
        mat.shadow_method = 'NONE'
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    bpy.data.objects['Lamp_Key'].hide_render = False
    bpy.data.objects['Lamp_Fill'].hide_render = False
    bpy.data.objects['Lamp_Back'].hide_render = False
    bpy.data.objects['Sun'].hide_render = False
    bpy.data.objects['Plane'].hide_render = False
    bpy.data.objects['Ground'].hide_render = False

    # Set the render settings back to what they were
    bpy.context.scene.display.render_aa = old_use_antialiasing
    bpy.data.scenes['Scene'].display.shading.light = old_view3DShading_light
    bpy.data.scenes['Scene'].display.shading.color_type = old_view3DShading_color_type
    # bpy.context.scene.render.film_transparent = old_transparency
    bpy.context.scene.node_tree.nodes['Denoise'].mute = old_denoise_enabled
    bpy.context.scene.view_settings.view_transform = old_view_transform
    bpy.context.scene.render.dither_intensity = old_dither
    render_args.filepath = old_filepath
    render_args.engine = old_engine

    return object_colors
