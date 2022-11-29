import numpy as np
import random  # if needed


def IsPointInsidePoly(vertex, poly_vertices):
    """
        Check if a point is inside a given polygon or not

        Parameters:
            vertex (list): [x,y] coordinates of the point
            poly_vertices (list): list of [x,y] coordinates of the polygon vertices

        Returns:
            bool: True if point is inside the polygon
    """
    n = poly_vertices.shape[0]
    result = False

    x = vertex[0]
    y = vertex[1]
    px1 = poly_vertices[0, 0]
    py1 = poly_vertices[0, 1]
    xints = x - 1.0
    for i in range(0, n + 1):
        px2 = poly_vertices[i % n, 0]
        py2 = poly_vertices[i % n, 1]
        if (min(py1, py2) < y):
            if (y <= max(py1, py2)):
                if (x <= max(px1, px2)):
                    if (py1 != py2):
                        xints = (y - py1) * (px2 - px1) / (py2 - py1) + px1
                    if (px1 == px2 or x <= xints):
                        result = not result
        px1 = px2
        py1 = py2

    return result

# lets create a function to create a square with rotation


def create_square2(x, y, size, rotation):
    # create a square
    square = np.array(
        [[x, y], [x + size, y], [x + size, y + size], [x, y + size]])
    # rotate the square
    rotation = np.deg2rad(rotation)
    square = np.array([rotate_point(square[0], square[i], rotation)
                      for i in range(square.__len__())])
    return square


def rotate_point(init_point, point, theta):
    if np.all(init_point == point):
        return point
    else:
        x0 = init_point[0]
        y0 = init_point[1]
        x_ = point[0]
        y_ = point[1]
        r0 = np.arctan2(y_ - y0, x_ - x0)
        r = theta
        r_ = r0 + r

        dist = np.sqrt((x0-x_)**2 + (y0-y_)**2)
        #x = x0 + dist / np.sqrt(1 + np.tan(r_)**2)
        x = x0 + dist * np.cos(r_)
        y = y0 + np.sqrt(dist**2 - (x-x0)**2)
        return [x, y]
# check if two squares intersect

# check if two squares intersect


def check_if_intersect(square1, square2):
    """
        Check if two squares intersect

        Parameters:
            square1 (np.array): 4x2 array of the vertices of the square
            square2 (np.array): 4x2 array of the vertices of the square

        Returns:
            bool: True if the squares intersect
    """
    # check if any of the points of square1 are inside square2
    for i in range(4):
        if IsPointInsidePoly(square1[i], square2):
            return True
    # check if any of the points of square2 are inside square1
    for i in range(4):
        if IsPointInsidePoly(square2[i], square1):
            return True
    return False


def check_if_intersect2(square1, square2):

    for i in range(4):
        if check_if_intersect_segment_square(square1[i], square1[(i + 1) % 4], square2):
            return True
        if check_if_intersect(square1, square2):
            return True


def check_if_intersect_segment_square(point1, point2, square):

    for i in range(4):
        if check_if_intersect_segment_segment(point1, point2, square[i], square[(i + 1) % 4]):
            return True
    return False


def check_if_intersect_segment_segment(point1, point2, point3, point4):

    if (point2[0] - point1[0]) * (point4[1] - point3[1]) - (point2[1] - point1[1]) * (point4[0] - point3[0]) == 0:
        return False
    # find the intersection point
    t = ((point1[0] - point3[0]) * (point3[1] - point4[1]) - (point1[1] - point3[1]) * (point3[0] - point4[0])) / (
        (point1[0] - point2[0]) * (point3[1] - point4[1]) - (point1[1] - point2[1]) * (point3[0] - point4[0]))
    u = -((point1[0] - point2[0]) * (point1[1] - point3[1]) - (point1[1] - point2[1]) * (point1[0] - point3[0])) / (
        (point1[0] - point2[0]) * (point3[1] - point4[1]) - (point1[1] - point2[1]) * (point3[0] - point4[0]))
    # check if the intersection point is on the segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        return True
    return False


def check_intersection(data, x, y, size, rotation):
    """
        Check if a square intersects with any of the squares in data

        Parameters:
            data (list): list of np.array of the vertices of the squares
            x (float): x coordinate of the center of the square
            y (float): y coordinate of the center of the square
            size (float): size of the square
            rotation (float): rotation of the square in degrees

        Returns:
            bool: True if the square intersects with any of the squares in data
    """
    # create a square
    square = create_square2(x, y, size, rotation)
    # check if it intersects with any of the squares in the data
    for i in range(data.__len__()):
        # create a square
        square2 = create_square2(
            data[i][0], data[i][1], data[i][2], data[i][3])
        # check if they intersect
        if check_if_intersect2(square, square2):
            return True
    return False
