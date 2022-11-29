import math
import time
from multiprocessing import Pool
from scipy.signal import argrelextrema
from sklearn.neighbors._kde import KernelDensity
import numpy as np
import random
import data as dt
import matplotlib.pyplot as plt
import geometry as geo
from rtree import index

# load the data from a file .npy
data = np.load('nmann.npy')
print(data.__len__())


def GetBandwidth(data):
    # get bandwidth for kernel density estimation
    # bw must be limited by:
    # 1. it minimizes the distance between the maximum and mean of a cluster
    # 2. the distance between two cluster means is maximazied
    # or be chosen manually
    # or be chosen by cross validation
    # or so that the cluster's sizes are not too different from each other
    bandwidth = 0

    return bandwidth


def rclustering(dataset, bandwidth_method='opt'):
    dataset = np.array(dataset)
    data = dataset[:, -1]
    if bandwidth_method == 'opt':
        bandwidth = GetBandwidth(data)
    else:
        bandwidth = float(bandwidth_method)
    #print("bandwidth: ", bandwidth)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
        data.reshape(-1, 1))
    s = np.linspace(0, 90)
    e = kde.score_samples(s.reshape(-1, 1))
    plt.plot(s, e)
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    # print them
    #print("minima: ", s[mi])
    #print("maxima: ", s[ma])
    # create a np.array of the clusters
    # maximum length is one more than minimum
    clusters = []
    for i in range(mi.__len__()):
        temp = []
        if i == 0:
            for j in range(data.__len__()):
                if data[j] <= s[mi[i]]:
                    temp.append(dataset[j])
            clusters.append(np.array(temp))
            continue
        if i > 0 and i <= mi.__len__()-1:
            for j in range(data.__len__()):
                if data[j] <= s[mi[i]] and data[j] > s[mi[i-1]]:
                    temp.append(dataset[j])
            clusters.append(np.array(temp))
            if i == mi.__len__()-1:
                temp = []
                for j in range(data.__len__()):
                    if data[j] > s[mi[i]]:
                        temp.append(dataset[j])
                clusters.append(np.array(temp))
            continue
    print("number of clusters: ", clusters.__len__())
    return np.array(clusters), s[mi], s[ma]

# compress data and calculate error for each cluster


def compress_data(clusters, min, max):
    print("No of clusters: ", clusters.__len__())
    print("No of min and max points: ", min.__len__(), max.__len__())
    error = [0 for i in range(clusters.__len__())]

    for i in range(clusters.__len__()):
        maxima = np.max(clusters[i][:, -1])
        print("maxima: ", maxima)
        mean = np.mean(clusters[i][:, -1])
        print("mean: ", mean)
        # calculate error
        for j in range(clusters[i].__len__()):
            error[i] += (maxima - clusters[i][j][-1])
            clusters[i][j][-1] = mean
    return clusters, error

# create a cost function for w_error and bandwidth, so that u choose bandwidth to minimize w_error
# w_error must take into  account the number of clusters, we dont want to have too many clusters or too few


def cost_function(bandwidth):
    clusters, min, max = rclustering(data, bandwidth_method=bandwidth)
    new_data, error = compress_data(clusters, min, max)
    print("error: ", error)
    w_error = 0
    n_error = []
    for i in range(error.__len__()):
        w_error += np.abs(error[i])
        n_error.append(error[i] / clusters[i].__len__())
    return w_error, n_error, error, clusters.__len__()


cluster, min_, max_ = rclustering(data, bandwidth_method=3)

clusters, error = compress_data(cluster, min_, max_)

# lets find the distance of a point from a square
# get distance of a point from a non axis aligned square


def dist_rsq(point, square):
    x0 = point[0]
    y0 = point[1]
    px0 = square[0]
    py0 = square[1]
    pl = square[2]
    theta = square[3]
    theta = np.deg2rad(theta)
    # rotate the point
    x = x0 * np.cos(theta) - y0 * np.sin(theta)
    y = x0 * np.sin(theta) + y0 * np.cos(theta)
    # rotate the square
    px = px0 * np.cos(theta) - py0 * np.sin(theta)
    py = px0 * np.sin(theta) + py0 * np.cos(theta)
    # find the distance
    dist = distance_from_square([x, y], [px, py, pl])
    return dist

# find the distance of a point from a square


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

# find the distance of a point from a square


def distance(point, square):
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


# we are gonna create an Rtree with the data as bounding boxes of ids and the data
idx = [None for _ in range(clusters.__len__())]
for i in range(clusters.__len__()):
    idx[i] = index.Index()
    data_new = arrange_data(clusters[i])

    for j in range(data_new.__len__()):
        # create tuple
        tup = (data_new[j][0], data_new[j][1], data_new[j][2], data_new[j][3])
        idx[i].insert(j, tup)
print(idx)


points = np.load('million_random_query_points.npy')
test = points[0:1]
N = 5000000


def cube(x):
    return math.sqrt(x)


if __name__ == "__main__":
    # first way, using multiprocessing
    start_time = time.perf_counter()
    with Pool() as pool:
        result = pool.map(NNRTREE, [idx[0], test[0], clusters[0], 1])
    finish_time = time.perf_counter()
    print("Program finished in {} seconds - using multiprocessing".format(finish_time-start_time))
    print("---")
    # second way, serial computation
    start_time = time.perf_counter()
    result = []
    for x in range(10, N):
        result.append(cube(x))
    finish_time = time.perf_counter()
    print("Program finished in {} seconds".format(finish_time-start_time))
