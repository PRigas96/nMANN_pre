import numpy as np
# lets find the distance of a point from a square
# get distance of a point from a non axis aligned square


def distance_from_square(point, square):
    x0 = point[0]
    y0 = point[1]
    px0 = square[0]
    py0 = square[1]
    pl = square[2]
    dist = 0
    distx = 0
    disty = 0
    # distance for l = 0
    dx = px0 - x0
    dy = py0 - y0
    # distance for l = pl
    dxl = dx + pl
    dyl = dy + pl
    cases = [dx > 0, dx < 0, dy > 0, dy < 0,
             dxl > 0, dxl < 0, dyl > 0, dyl < 0]
    # check x axis first
    if dxl < 0:
        distx = abs(dxl)
    elif dx > 0:
        distx = abs(dx)
    else:
        distx = 0

    # check y axis
    if dyl < 0:
        disty = abs(dyl)
    elif dy > 0:
        disty = abs(dy)
    else:
        disty = 0

    dist = np.sqrt(distx**2 + disty**2)
    return dist


def rotate_square(square):
    pl = square[-2]
    theta = np.deg2rad(square[-1])
    pl_new = pl * np.cos(theta)
    return [square[0], square[1], pl_new, square[-1]]
# get deg of 2 points


def get_deg(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    if x1 == x2:
        if y1 > y2:
            return 90
        else:
            return 270
    else:
        m = (y2-y1)/(x2-x1)
        #print("m: ", m)
        deg = np.rad2deg(np.arctan(m))
        if x1 > x2:
            deg += 180
        elif y1 > y2:
            deg += 360
        return deg


def bf_nn(data, point):
    min_dist = 100000
    min_index = 0
    for i in range(data.__len__()):
        dist = dist_rsq(point, data[i])
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return min_index, min_dist


def dist_rsq(point, square):
    # rotate the point
    point = rotate_point([square[0], square[1]], point, np.deg2rad(square[-1]))
    #print("point: ", point)
    dist = distance_from_square(point, square)
    return dist


def rotate_point(init_point, point, theta):
    if np.all(init_point == point):
        return point

    else:
        x0 = init_point[0]
        y0 = init_point[1]
        x1 = point[0]
        y1 = point[1]
        theta_0 = np.deg2rad(get_deg(init_point, point))
        theta_r = theta
        theta_1 = theta_0 - theta_r
        #print("theta_0: ", theta_0)
        #print("theta_r: ", theta_r)
        #print("theta_1: ", theta_1)
        dist = np.sqrt((x0-x1)**2 + (y0-y1)**2)
        pr = np.rad2deg(theta_1)
        if pr >= 0 and pr <= 90:
            prx = 1
            pry = 1
        elif pr > 90 and pr <= 180:
            prx = -1
            pry = 1
        elif pr > 180 and pr <= 270:
            prx = -1
            pry = -1
        else:
            prx = 1
            pry = -1

        #print("dist: ", dist)
        x1_r = x0 + prx * dist/(np.sqrt(1+np.tan(theta_1)**2))
        #print("x1_r: ", x1_r)

        y1_r = y0 + pry * np.sqrt(dist**2 - (x1_r-x0)**2)
        #print("y1_r: ", y1_r)
        return [x1_r, y1_r]


def bf_nn(data, point):
    min_dist = 100000
    min_index = 0
    for i in range(data.__len__()):
        dist = dist_rsq(point, data[i])
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return min_index, min_dist


def create_points(num, x0, y0):
    points = []
    for i in range(num):
        x = np.random.randint(x0[0], x0[1])
        y = np.random.randint(y0[0], y0[1])
        points.append([x, y])
    return points


def arrange_data(data):
    tup = []
    for i in range(data.__len__()):
        tup.append((data[i][0], data[i][1], data[i][0] +
                   data[i][2], data[i][1]+data[i][2]))

    return tup


def NNRTREE(idx, point, data, k):
    result = list(idx.nearest((point[0], point[1], point[0], point[1]), k))
    sq = [data[result[i]] for i in range(result.__len__())]
    dist = [dist_rsq(point, data[result[i]]) for i in range(sq.__len__())]

    #sq = []
    #dist = []
    # for i in range(result.__len__()):
    #    sq.append(data[result[i]])
    #    dist.append(dist_rsq(point, data[result[i]]))
    #dist = distance(point, sq)
    return result, dist, sq
