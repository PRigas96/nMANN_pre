from utils import nnrtree as nnrt
from rtree import index
import numpy as np


def index_rtree(data):
    idx = [None for _ in range(data.__len__())]
    for i in range(data.__len__()):
        idx[i] = index.Index()
        data_new = nnrt.arrange_data(data[i])
        for j in range(data_new.__len__()):
            # create tuple
            tup = (data_new[j][0], data_new[j][1],
                   data_new[j][2], data_new[j][3])
            idx[i].insert(j, tup)
    return idx


def choose_data(choice='large'):
    if choice == 'small':
        points_tot = np.load(
            'source/query_points/million_random_query_points.npy')
        sq_bf = np.load("source/bruteforce_results/sq_bf.npy")
        data = np.load('source/datasets/nmann.npy')
    else:
        points_tot = np.load('source/query_points/qp10k.npy')
        sq_bf = np.load('source/bruteforce_results/sq_bf10k.npy')
        data = np.load('source/datasets/nmann10k.npy')
    print("There are {} polygons in the dataset".format(data.shape[0]))
    print("There are {} [x,y] points".format(points_tot.shape[0]))
    print("There are {} square objects as results".format(sq_bf.shape[0]))
    return data, points_tot, sq_bf

# def get_acc(idx, data, points_tot, sq_bf):
