import math
import random
from matplotlib import patches
import numpy as np


def distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance
    """
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def get_distance_from_point_to_line(point, line_point1, line_point2):
    # When the coordinates of two points are the same point, the distance between the return point and the point
    if line_point1 == line_point2:
        point_array = np.array(point)
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array - point1_array)
    # Calculate the three parameters of a straight line
    a = line_point2[1] - line_point1[1]
    b = line_point1[0] - line_point2[0]
    c = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    # Calculate the distance based on the distance formula from the point to the straight line
    return np.abs(a * point[0] + b * point[1] + c) / (np.sqrt(a ** 2 + b ** 2))


class Vector:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Vector({self.x}, {self.y})'

    def get_magnitude(self):
        """
        Get vector size
        """
        return math.sqrt(self.x * self.x + self.y * self.y)

    def __add__(self, other):
        """
        Vector addition
        """
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """
        Vector subtraction
        """
        return Vector(self.x - other.x, self.y - other.y)

    def dot_product(self, other):
        """
        Dot multiplication
        """
        return self.x * other.x + self.y * other.y

    def edge(self, other):
        """
        Two-point generating edge
        """
        return self - other

    def perpendicular(self):
        """
        Vertical projection
        """
        return Vector(self.y, 0 - self.x)

    def normalize(self):
        """
        Unitization
        """
        v = Vector(0, 0)
        m = self.get_magnitude()
        if m != 0:
            v.x = self.x / m
            v.y = self.y / m
        return v

    def normal(self):
        """
        Unit vector of the projection axis
        """
        return self.perpendicular().normalize()


class Projection:

    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def __repr__(self):
        return f'Projection({self.min}, {self.max})'

    def overlaps(self, other, min_dist):
        return self.max > other.min - min_dist and other.max > self.min - min_dist

    def contains(self, other):
        return self.max > other.max and self.min < other.min


class Polygon:

    def __init__(self, points):
        self.points = points
        self.objs = []
        self.fc = None
        self.ec = None
        self.lw = None
        self.shape_3d = None
        self.color = None
        self.material = ''
        self.area_ratio = 0
        self.index = 0
        self.adj = {'left': [], 'right': [], 'up': [], 'down': []}
        self.size = ''

    def coordinate_points(self):
        return self.points

    def to_patch(self):
        return patches.Polygon(self.points, fc=self.fc, ec=self.ec, lw=self.lw)

    def get_adj(self, shapes):
        """
        Generate adjacency list
        """
        for shape in shapes:
            if shape.index == self.index:
                continue
            center1_x, center1_y = self.center()
            center2_x, center2_y = shape.center()
            if center1_x < center2_x:
                self.adj['right'].append(shape.index)
            if center1_x > center2_x:
                self.adj['left'].append(shape.index)
            if center1_y < center2_y:
                self.adj['up'].append(shape.index)
            if center1_y > center2_y:
                self.adj['down'].append(shape.index)

    def area(self):
        area_ = 0
        q = self.points[-1]
        for p in self.points:
            area_ += p[0] * q[1] - p[1] * q[0]
            q = p
        return math.fabs(area_) / 2

    def get_axes(self):
        """
        Get the projection axis
        """
        v1 = Vector(0, 0)
        v2 = Vector(0, 0)
        axes = []
        for i in range(len(self.points) - 1):
            v1.x = self.points[i][0]
            v1.y = self.points[i][1]
            v2.x = self.points[i + 1][0]
            v2.y = self.points[i + 1][1]
            axes.append(v1.edge(v2).normal())
        v1.x = self.points[len(self.points) - 1][0]
        v1.y = self.points[len(self.points) - 1][1]
        v2.x = self.points[0][0]
        v2.y = self.points[0][1]
        axes.append(v1.edge(v2).normal())
        return axes

    def project(self, axis):
        """
        projection
        """
        scalars = []
        v = Vector(0, 0)
        for i in range(len(self.points)):
            v.x = self.points[i][0]
            v.y = self.points[i][1]
            scalars.append(v.dot_product(axis))
        return Projection(min(scalars), max(scalars))

    def separation_on_axes(self, axes, other, min_dist):
        for i in range(len(axes)):
            axis = axes[i]
            projection1 = other.project(axis)
            projection2 = self.project(axis)
            if not projection1.overlaps(projection2, min_dist):
                return True
        return False

    def detect_collision(self, other, min_dist):
        if isinstance(other, Circle):
            return other.detect_collision(self, min_dist)
        else:
            axes = other.get_axes()
            axes.extend(self.get_axes())
            return not self.separation_on_axes(axes, other, min_dist)

    def center(self):
        x, y = 0, 0
        for p in self.points:
            x += p[0]
            y += p[1]
        x = x / float(len(self.points))
        y = y / float(len(self.points))
        return x, y

    def bbox(self):
        x_min = min([p[0] for p in self.points])
        x_max = max([p[0] for p in self.points])
        y_min = min([p[1] for p in self.points])
        y_max = max([p[1] for p in self.points])
        return x_min, x_max, y_min, y_max

    def separation_on_axes_contains(self, axes, other):
        for i in range(len(axes)):
            axis = axes[i]
            projection1 = self.project(axis)
            projection2 = other.project(axis)
            if not projection1.contains(projection2):
                return True
        return False

    def in_polygon(self, point):
        px, py = point
        is_in = False
        for i, corner in enumerate(self.points):
            next_i = i + 1 if i + 1 < len(self.points) else 0
            x1, y1 = corner
            x2, y2 = self.points[next_i]
            if (x1 == px and y1 == py) or (x2 == px and y2 == py):
                is_in = True
                break
            if min(y1, y2) < py <= max(y1, y2):
                x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if x == px:
                    is_in = True
                    break
                elif x > px:
                    is_in = not is_in
        return is_in

    def cover(self, shape):
        if isinstance(shape, Circle):
            ds = []
            q = self.points[-1]
            for p in self.points:
                d1 = get_distance_from_point_to_line([shape.x, shape.y], p, q)
                d2 = distance(shape.x, shape.y, q[0], q[1])
                ds.append(d1)
                ds.append(d2)
                q = p
            if not self.in_polygon(shape.center()):
                return False
            return min(ds) > shape.r
        else:
            axes = shape.get_axes()
            axes.extend(self.get_axes())
            return not self.separation_on_axes_contains(axes, shape)

    def check_margin_x(self, shape, margin_x_rate):
        if isinstance(shape, Circle):
            return shape.check_margin_x(self, margin_x_rate)
        else:
            min_x1 = min([shape.points[i][0] for i in range(len(shape.points))])
            max_x1 = max([shape.points[i][0] for i in range(len(shape.points))])
            min_x = min([self.points[i][0] for i in range(len(self.points))])
            max_x = max([self.points[i][0] for i in range(len(self.points))])
            x1, y1 = shape.center()
            x, y = self.center()
            if min_x < max_x1 and max_x > min_x1:
                if max_x1 < x:
                    if max_x1 - min_x < margin_x_rate * min(x - min_x, max_x1 - x1):
                        return True
                elif min_x1 > x:
                    if max_x - min_x1 < margin_x_rate * min(max_x - x, x1 - min_x1):
                        return True
                return False
            else:
                return True

    def check_margin_y(self, shape, margin_y_rate):
        if isinstance(shape, Circle):
            return shape.check_margin_y(self, margin_y_rate)
        else:
            min_y1 = min([shape.points[i][1] for i in range(len(shape.points))])
            max_y1 = max([shape.points[i][1] for i in range(len(shape.points))])
            min_y = min([self.points[i][1] for i in range(len(self.points))])
            max_y = max([self.points[i][1] for i in range(len(self.points))])
            x1, y1 = shape.center()
            x, y = self.center()
            if min_y < max_y1 and max_y > min_y1:
                if max_y1 < y:
                    if max_y1 - min_y < margin_y_rate * min(y - min_y, max_y1 - y1):
                        return True
                elif min_y1 > y:
                    if max_y - min_y1 < margin_y_rate * min(max_y - y, y1 - min_y1):
                        return True
                return False
            else:
                return True


class Triangle(Polygon):

    def __init__(self, x1, y1, x2, y2, x3, y3):
        super().__init__([[x1, y1], [x2, y2], [x3, y3]])
        self.shape = 'triangle'
        self.rotation = 0

    def rotate(self, rotation):
        self.rotation = rotation
        radians = math.radians(rotation)
        for i in range(1, 3):
            x, y = self.points[i][0], self.points[i][1]
            self.points[i][0] = ((x - self.points[0][0]) * math.cos(radians) +
                                 (y - self.points[0][1]) * math.sin(radians) + self.points[0][0])
            self.points[i][1] = ((y - self.points[0][1]) * math.cos(radians) -
                                 (x - self.points[0][0]) * math.sin(radians) + self.points[0][1])

    def __repr__(self):
        return f'Triangle({self.points[0][0]}, {self.points[0][1]}, ' \
               f'{self.points[1][0]}, {self.points[1][1]}, {self.points[2][0]}, {self.points[2][1]})'

    def possible(self, w, h, area_max, area_min, min_angle, dist_to_axis=0):
        """
        whether feasible
        """
        area = self.area()
        if area > area_max or area < area_min:
            return False
        self.area_ratio = area / (w * h)
        # Check whether the angle meets the requirements
        angles = []
        a = distance(self.points[0][0], self.points[0][1], self.points[1][0], self.points[1][1])
        b = distance(self.points[1][0], self.points[1][1], self.points[2][0], self.points[2][1])
        c = distance(self.points[0][0], self.points[0][1], self.points[2][0], self.points[2][1])
        angles.append(math.asin(round(2 * area / (a * b), 4)))
        angles.append(math.asin(round(2 * area / (b * c), 4)))
        angles.append(math.asin(round(2 * area / (a * c), 4)))
        if min(angles) < math.radians(min_angle):
            return False
        if min([self.points[i][0] for i in range(len(self.points))]) < dist_to_axis:
            return False
        elif max([self.points[i][0] for i in range(len(self.points))]) > w:
            return False
        elif min([self.points[i][1] for i in range(len(self.points))]) < dist_to_axis:
            return False
        elif max([self.points[i][1] for i in range(len(self.points))]) > h:
            return False
        return True


class Rectangle(Polygon):

    def __init__(self, x1, y1, w, h, rotation):
        self.points = [[x1, y1]]
        self.w = w
        self.h = h
        self.rotation = rotation
        self.radians = math.radians(rotation)
        self.points.append(self.rotate(self.points[0][0], self.points[0][1] + h))
        self.points.append(self.rotate(self.points[0][0] + w, self.points[0][1] + h))
        self.points.append(self.rotate(self.points[0][0] + w, self.points[0][1]))
        super().__init__(self.points)
        self.shape = 'rectangle'
        self.rotation = rotation

    def __repr__(self):
        return f'Rectangle({self.points[0][0]}, {self.points[0][1]}, {self.w}, {self.h}, {self.rotation})'

    def rotate(self, x, y):
        """
        Calculate the coordinates after rotation
        """
        return [(x - self.points[0][0]) * math.cos(self.radians) +
                (y - self.points[0][1]) * math.sin(self.radians) + self.points[0][0],
                (y - self.points[0][1]) * math.cos(self.radians) -
                (x - self.points[0][0]) * math.sin(self.radians) + self.points[0][1]]

    def possible(self, w, h, area_max, area_min, dist_to_axis=0):
        """
        If feasible
        """
        if self.area() > area_max or self.area() < area_min:
            return False
        self.area_ratio = self.area() / (w * h)
        if min([self.points[i][0] for i in range(4)]) < dist_to_axis:
            return False
        elif max([self.points[i][0] for i in range(4)]) > w:
            return False
        elif min([self.points[i][1] for i in range(4)]) < dist_to_axis:
            return False
        elif max([self.points[i][1] for i in range(4)]) > h:
            return False
        return True


class Circle:

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        self.objs = []
        self.shape = 'circle'
        self.shape_3d = None
        self.fc = None
        self.ec = None
        self.lw = None
        self.color = None
        self.adj = {'left': [], 'right': [], 'up': [], 'down': []}
        self.material = ''
        self.area_ratio = 0
        self.index = 0
        self.size = ''
        self.rotation = 0

    def coordinate_points(self):
        return [[self.x, self.y], self.r]

    def __repr__(self):
        return f'Circle(({self.x}, {self.y}), {self.r})'

    def get_adj(self, shapes):
        """
        Generate adjacency list
        """
        for shape in shapes:
            if shape.index == self.index:
                continue
            center1_x, center1_y = self.center()
            center2_x, center2_y = shape.center()
            if center1_x < center2_x:
                self.adj['right'].append(shape.index)
            if center1_x > center2_x:
                self.adj['left'].append(shape.index)
            if center1_y < center2_y:
                self.adj['up'].append(shape.index)
            if center1_y > center2_y:
                self.adj['down'].append(shape.index)

    def to_patch(self):
        """
        Drawing object
        """
        return patches.Circle((self.x, self.y), self.r, fc=self.fc, ec=self.ec, lw=self.lw)

    def center(self):
        return self.x, self.y

    def possible(self, w, h, area_max, area_min, dist_to_axis=0):
        """
        If feasible
        """
        if self.area() > area_max or self.area() < area_min:
            return False
        self.area_ratio = self.area() / (w * h)
        if self.x - self.r < dist_to_axis:
            return False
        elif self.x + self.r > w:
            return False
        elif self.y - self.r < dist_to_axis:
            return False
        elif self.y + self.r > h:
            return False
        return True

    def area(self):
        return math.pi * self.r * self.r

    def project(self, axis):
        scalars = []
        dot_pro = Vector(self.x, self.y).dot_product(axis)
        scalars.append(dot_pro)
        scalars.append(dot_pro + self.r)
        scalars.append(dot_pro - self.r)
        return Projection(min(scalars), max(scalars))

    def get_closest_point(self, other):
        """
        Get the closest point of the polygon to the circle
        """
        min_dis = float('inf')
        closest_x, closest_y = 0, 0
        for i in range(len(other.points)):
            x = other.points[i][0]
            y = other.points[i][1]
            dis = distance(x, y, self.x, self.y)
            if dis < min_dis:
                min_dis = dis
                closest_x = x
                closest_y = y
        return closest_x, closest_y

    def detect_collision(self, other, min_dist):
        # Collision with circle detection
        if isinstance(other, Circle):
            return distance(self.x, self.y, other.x, other.y) < self.r + other.r + min_dist
        # Detect collision with polygon
        else:
            closest_x, closest_y = self.get_closest_point(other)
            v1 = Vector(self.x, self.y)
            v2 = Vector(closest_x, closest_y)
            axes = other.get_axes()
            axes.append((v1 - v2).normalize())
            return not other.separation_on_axes(axes, self, min_dist)

    def bbox(self):
        x_min = self.x - self.r
        x_max = self.x + self.r
        y_min = self.y - self.r
        y_max = self.y + self.r
        return x_min, x_max, y_min, y_max

    def cover(self, shape):
        if isinstance(shape, Circle):
            if distance(shape.x, shape.y, self.x, self.y) > self.r - shape.r:
                return False
        else:
            for p in shape.points:
                if distance(p[0], p[1], self.x, self.y) > self.r:
                    return False
        return True

    def check_margin_x(self, shape, margin_x_rate):
        if isinstance(shape, Circle):
            min_x1 = shape.x - shape.r
            max_x1 = shape.x + shape.r
        else:
            min_x1 = min([shape.points[i][0] for i in range(len(shape.points))])
            max_x1 = max([shape.points[i][0] for i in range(len(shape.points))])
        x1, y1 = shape.center()
        min_x = self.x - self.r
        max_x = self.x + self.r
        if min_x < max_x1 and max_x > min_x1:
            if max_x1 < self.x:
                if max_x1 - min_x < margin_x_rate * min(self.r, max_x1 - x1):
                    return True
            elif min_x1 > self.x:
                if max_x - min_x1 < margin_x_rate * min(self.r, x1 - min_x1):
                    return True
            return False
        else:
            return True

    def check_margin_y(self, shape, margin_y_rate):
        if isinstance(shape, Circle):
            min_y1 = shape.y - shape.r
            max_y1 = shape.y + shape.r
        else:
            min_y1 = min([shape.points[i][1] for i in range(len(shape.points))])
            max_y1 = max([shape.points[i][1] for i in range(len(shape.points))])
        x1, y1 = shape.center()
        min_y = self.y - self.r
        max_y = self.y + self.r
        if min_y < max_y1 and max_y > min_y1:
            if max_y1 < self.y:
                if max_y1 - min_y < margin_y_rate * min(self.r, max_y1 - y1):
                    return True
            elif min_y1 > self.y:
                if max_y - min_y1 < margin_y_rate * min(self.r, y1 - min_y1):
                    return True
            return False
        else:
            return True


def get_obj(materials, color, color_rgb, shape, size, center, rotation, r_list, shape_3d):
    # factor = math.sqrt(zoom)
    if shape == 'triangle':
        # if size == 1:
        #     r = r * factor
        r = r_list[size]
        x1 = center[0]
        y1 = center[1] + r
        x2 = center[0] - math.cos(math.radians(30)) * r
        y2 = center[1] - math.sin(math.radians(30)) * r
        x3 = center[0] + math.cos(math.radians(30)) * r
        y3 = y2
        obj = Triangle(x1, y1, x2, y2, x3, y3)
        obj.rotate(rotation)
    elif shape == 'rectangle':
        # if size == 0:
        #     w = r * math.sqrt(2)
        #     h = w
        # else:
        #     w = r * math.sqrt(2) * factor
        #     h = w
        w = r_list[size] * math.sqrt(2)
        h = w
        obj = Rectangle(center[0], center[1], w, h, rotation)
    else:
        # if size == 1:
        #     r = r * factor
        r = r_list[size]
        obj = Circle(center[0], center[1], r)
    obj.material = random.choice(materials)
    obj.size = size
    obj.fc = tuple(color_rgb)
    obj.color = color
    obj.shape_3d = shape_3d
    return obj

# def check_min_dist(shape1, shape2, min_dist, r):
#     center1 = shape1.center()
#     center2 = shape2.center()
#     size1 = 1 if shape1.size == 'small' else 2
#     size2 = 1 if shape1.size == 'small' else 2
#     if distance(center1[0], center1[1], center2[0], center2[1]) < min_dist + r*(size1+size2):
#         return False
#     return True


def check(shapes, shape, w, h, area_max, area_min, min_dist=20, margin_x=30, margin_y=30, min_angle=25, dist_to_axis=2):
    # If feasible
    if isinstance(shape, Triangle):
        if not shape.possible(w, h, area_max, area_min, min_angle, dist_to_axis):
            return False
    else:
        if not shape.possible(w, h, area_max, area_min, dist_to_axis):
            return False
    # Center distance
    # for s in shapes:
    #     if not check_min_dist(s, shape, min_dist):
    #         return False
    # for s in shapes:
    #     if not shape.check_margin_x(s, margin_x_rate) or not shape.check_margin_y(s, margin_y_rate):
    #         return False
    x1, y1 = shape.center()
    for s in shapes:
        x2, y2 = s.center()
        if abs(x1 - x2) < margin_x:
            return False
        if abs(y1 - y2) < margin_y:
            return False
        # Whether overlap
        if s.detect_collision(shape, min_dist):
            return False
    return True


# def check_for_obj(shapes, shape, w, h, area_max, area_min, min_dist=20, margin_x_rate=0.5, margin_y_rate=0.5,
#                   min_angle=25):
#     # If feasible
#     if isinstance(shape, Triangle):
#         if not shape.possible(w, h, area_max, area_min, min_angle):
#             return False
#     else:
#         if not shape.possible(w, h, area_max, area_min):
#             return False
#     Center distance
#     for s in shapes:
#         if not check_min_dist(s, shape, min_dist, r):
#             logger.debug('min dist fail')
#             return False

def check_for_obj(shapes, shape, min_dist=20, margin_x_rate=0.5, margin_y_rate=0.5):
    for s in shapes:
        # Whether overlap
        if not shape.check_margin_x(s, margin_x_rate) or not shape.check_margin_y(s, margin_y_rate):
            return False
        if s.detect_collision(shape, min_dist):
            return False
    return True


def check_po(shapes, shape, w, h, area_max, area_min, min_dist=20, min_angle=25):
    # If feasible
    if isinstance(shape, Triangle):
        if not shape.possible(w, h, area_max, area_min, min_angle):
            return False
    else:
        if not shape.possible(w, h, area_max, area_min):
            return False
    for s in shapes:
        if s.detect_collision(shape, min_dist):
            return False
    return True


def choose_and_remove(items):
    # pick an item index
    if items:
        index = random.randrange(len(items))
        return items.pop(index)
    return None


def gen_planes(args, properties):
    print('# Generating planes...')
    # Number of random plane graphics
    num_shapes = random.randint(1, len(args.plane_area_rate))

    # Area range of individual geometry plane
    area_max_rate = args.plane_area_rate[num_shapes - 1][1]
    area_min_rate = args.plane_area_rate[num_shapes - 1][0]
    img_width = properties["plane_size"]["width"] * args.ratio_of_2d_to_3d - args.dist_to_axis
    img_length = properties["plane_size"]["length"] * args.ratio_of_2d_to_3d - args.dist_to_axis
    plane_shapes = properties["plane_shapes"]
    area_max = (img_length - args.dist_to_axis) * img_width * area_max_rate
    area_min = (img_length - args.dist_to_axis) * img_width * area_min_rate

    count_total = 0
    shapes = []
    best_shapes_result = []

    while len(shapes) < num_shapes:
        if len(best_shapes_result) <= len(shapes):
            best_shapes_result = shapes.copy()
        if count_total > args.max_plane_retries:
            shapes = best_shapes_result.copy()
            break
        count = 0
        untried_shapes = plane_shapes.copy()
        shape = choose_and_remove(untried_shapes)
        while True:
            if count_total > args.max_plane_retries:
                break
            if count > int(args.max_plane_retries / (num_shapes * len(plane_shapes))):
                if len(untried_shapes):
                    shape = choose_and_remove(untried_shapes)
                    count = 0
                elif len(shapes):
                    shapes.sort(key=lambda x_: x_.area())
                    shapes.pop()
                    break
                else:
                    break
            count += 1
            count_total += 1
            if shape == 'triangle':
                xs = [random.randint(args.dist_to_axis, img_length)]
                for _ in range(2):
                    xs.append(random.randint(int(xs[0] - 5 * img_width * area_max_rate),
                                             int(xs[0] + 6 * img_length * area_max_rate)))
                ys = [random.randint(args.dist_to_axis, img_width)]
                for _ in range(2):
                    ys.append(random.randint(int(ys[0] - 5 * img_width * area_max_rate),
                                             int(ys[0] + 6 * img_width * area_max_rate)))
                new_shape = Triangle(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2])
            elif shape == 'rectangle':
                x = random.randint(args.dist_to_axis, img_length)
                y = random.randint(args.dist_to_axis, img_width)
                rect_wh_rate = random.uniform(args.min_rect_asp_rate, 1.01)
                w_ = random.randint(int(math.sqrt((area_min / rect_wh_rate))),
                                    int(math.sqrt((area_max / rect_wh_rate))))
                h_ = w_ * rect_wh_rate
                r = random.randint(0, 180)
                new_shape = Rectangle(x, y, w_, h_, r)
            else:
                x = random.randint(args.dist_to_axis, img_length)
                y = random.randint(args.dist_to_axis, img_width)
                r = random.randint(int(math.sqrt((area_min / math.pi))), int(math.sqrt((area_max / math.pi))))
                new_shape = Circle(x, y, r)
            # Check whether the generated graphics meet the requirements
            if check(shapes, new_shape, img_length, img_width, area_max, area_min, args.min_dist_plane,
                     args.margin_x_plane, args.margin_y_plane, args.min_angle, args.dist_to_axis):
                shapes.append(new_shape)
                break
            else:
                continue

    # Sort by plane's area ratio
    shapes.sort(key=lambda x_: x_.area_ratio)
    index = 0
    for shape in shapes:
        shape.index = index
        index += 1
    print(f"  Randomly generated number: {num_shapes}")
    print(f"  Actual generated quantity: {len(shapes)}. Tried  {count_total} times")

    return shapes


def gen_objs_in_plane(
        args,
        properties,
        obj_2d_shapes,
        obj_3d_shapes,
        shapes,
        obj_materials,
        obj_colors
):
    obj_index = 0
    print(f'# The 3d shapes of the objects to be generated: {obj_3d_shapes}')
    print(f'# The 2d shapes of the objects to be generated: {obj_2d_shapes}')
    print(f'# The colors of the objects to be generated: {obj_colors}')
    print(f'# The materials of the objects to be generated: {obj_materials}')
    print('# Generating objects inside the planes...')
    unique_2d_objs = list(set(obj_2d_shapes))
    objs_sizes = [*properties["sizes"]]
    planes_objs = []
    none_obj_plane = []
    for shape in shapes:
        x_min, x_max, y_min, y_max = shape.bbox()
        count_0 = 0
        num_objs = random.randint(args.min_num_obj, args.max_num_obj)
        best_objs_result = []
        best_planes_objs_result = []
        while len(shape.objs) < num_objs:
            if len(best_objs_result) <= len(shape.objs):
                best_objs_result = shape.objs.copy()
                best_planes_objs_result = planes_objs.copy()
            if count_0 > args.max_obj_retries:
                shape.objs = best_objs_result.copy()
                planes_objs = best_planes_objs_result.copy()
                break
            count_1 = 0
            untried_objs = unique_2d_objs.copy()
            s = choose_and_remove(untried_objs)
            untried_sizes = objs_sizes.copy()
            size_0 = choose_and_remove(untried_sizes)
            size = size_0
            while True:
                if count_0 > args.max_obj_retries:
                    break
                # You don’t need to look for a large one if you haven’t found a small-sized object.
                if len(untried_sizes) and size != objs_sizes[-1] and \
                        count_1 > int(args.max_obj_retries / (num_objs * len(unique_2d_objs) * len(objs_sizes))):
                    size = choose_and_remove(untried_sizes)
                elif count_1 > int(args.max_obj_retries / (num_objs * len(unique_2d_objs))):
                    if len(untried_objs):
                        untried_sizes = objs_sizes.copy()
                        # Keep it the same size that you haven't found at first.
                        size = size_0
                        untried_sizes.remove(size)
                        # size = choose_and_remove(untried_sizes)
                        s = choose_and_remove(untried_objs)
                        count_1 = 0
                    elif len(shape.objs):
                        # shape.objs.sort(key=lambda x_: x_.area())
                        temp_obj = shape.objs.pop()
                        planes_objs.remove(temp_obj)
                        break
                    else:
                        break
                count_1 += 1
                count_0 += 1
                center_x = random.randint(int(x_min), int(x_max))
                center_y = random.randint(int(y_min), int(y_max))
                rotation = random.randint(0, 180)
                obj_color = random.choice(obj_colors)
                obj_color_rgb = [float(x) / 255.0 for x in properties['obj_colors'][obj_color]]
                s_index = [i for i, e in enumerate(obj_2d_shapes) if e == s]
                s_3d = random.choice([obj_3d_shapes[index] for index in s_index])
                new_shape = get_obj(obj_materials, obj_color, obj_color_rgb, s, size, [center_x, center_y], rotation,
                                    properties['sizes'], shape_3d=s_3d)
                if shape.cover(new_shape):
                    # has_collision = False
                    # for obj in shape.objs:
                    #     if obj.detect_collision(new_shape, 0):
                    #         has_collision = True
                    #         break
                    if check_for_obj(planes_objs, new_shape, args.min_dist_obj, args.margin_x_obj_rate,
                                     args.margin_y_obj_rate):
                        shape.objs.append(new_shape)
                        planes_objs.append(new_shape)
                        break
                    else:
                        continue
                else:
                    continue
        if len(shape.objs) == 0:
            none_obj_plane.append(shape)
        else:
            for obj in shape.objs:
                obj.index = obj_index
                obj_index += 1

        print(f"  Randomly generated number: {num_objs}")
        print(f"  Actual generated quantity: {len(shape.objs)}. Tried {count_0} times ")
        print(f"  Current total number of objects inside all the planes: {len(planes_objs)}")

    if len(none_obj_plane) != 0:
        for p in none_obj_plane:
            shapes.remove(p)
            print("Removed the plane contains no objects.")
        index = 0
        for shape in shapes:
            shape.index = index
            index += 1

    return obj_index, planes_objs, shapes


def gen_objs_in_ng_plane(
        args,
        properties,
        shapes,
        obj_2d_shapes,
        obj_3d_shapes,
        obj_index,
        obj_materials,
        obj_colors,
        planes_objs
):
    print('# Generating objects outside the planes...')
    num_objs = random.randint(args.min_num_obj_ng, args.max_num_obj_ng)
    unique_2d_objs = list(set(obj_2d_shapes))
    objs_sizes = [*properties["sizes"]]
    img_width = properties["plane_size"]["width"] * args.ratio_of_2d_to_3d
    img_length = properties["plane_size"]["length"] * args.ratio_of_2d_to_3d
    bg_objs = []
    shapes_1 = shapes.copy()
    count_3 = 0
    best_bg_objs_result = []
    while len(bg_objs) < num_objs:
        if len(best_bg_objs_result) <= len(bg_objs):
            best_bg_objs_result = bg_objs.copy()
        if count_3 > args.max_obj_retries:
            bg_objs = best_bg_objs_result.copy()
            break
        count_4 = 0
        untried_objs = unique_2d_objs.copy()
        s = choose_and_remove(untried_objs)
        untried_sizes = objs_sizes.copy()
        size_0 = choose_and_remove(untried_sizes)
        size = size_0
        while True:
            if count_3 > args.max_obj_retries:
                break
            # You don’t need to look for a large one if you haven’t found a small-sized object.
            if len(untried_sizes) and size != objs_sizes[-1] and \
                    count_4 > int(args.max_obj_retries / (num_objs * len(unique_2d_objs) * len(objs_sizes))):
                size = choose_and_remove(untried_sizes)
            elif count_4 > int(args.max_obj_retries / (num_objs * len(unique_2d_objs))):
                if len(untried_objs):
                    untried_sizes = objs_sizes.copy()
                    # Keep it the same size that you haven't found at first.
                    size = size_0
                    untried_sizes.remove(size)
                    # size = choose_and_remove(untried_sizes)
                    s = choose_and_remove(untried_objs)
                    count_4 = 0
                elif len(bg_objs):
                    bg_objs.pop()
                    shapes_1.pop()
                    obj_index -= 1
                    break
                else:
                    break
            count_4 += 1
            count_3 += 1
            center_x = random.randint(0, img_length)
            center_y = random.randint(0, img_width)
            rotation = random.randint(0, 180)
            obj_color = random.choice(obj_colors)
            obj_color_rgb = [float(x) / 255.0 for x in properties['obj_colors'][obj_color]]
            s_index = [i for i, e in enumerate(obj_2d_shapes) if e == s]
            s_3d = random.choice([obj_3d_shapes[index] for index in s_index])
            new_shape = get_obj(obj_materials, obj_color, obj_color_rgb, s, size, [center_x, center_y], rotation,
                                properties['sizes'], shape_3d=s_3d)
            if check_po(shapes_1, new_shape, img_length, img_width, img_length * img_width, 0, args.min_dist_obj) \
                    and check_for_obj([*planes_objs, *bg_objs], new_shape, args.min_dist_obj, args.margin_x_obj_rate,
                                      args.margin_y_obj_rate):
                new_shape.index = obj_index
                obj_index += 1
                shapes_1.append(new_shape)
                bg_objs.append(new_shape)
                break
            else:
                continue
    print(f"  Randomly generated number: {num_objs}")
    print(f"  Actual generated quantity: {len(bg_objs)}. Tried {count_3} times")

    return bg_objs
