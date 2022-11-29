from sklearn.neighbors._kde import KernelDensity
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt


def GetBandwidth(data):
    # get bandwidth for kernel density estimation
    # bw must be limited by:
    # 1. it minimizes the distance between the maximum and mean of a cluster
    # 2. the distance between two cluster means is maximazied
    # or be chosen manually
    # or be chosen by cross validation
    # or so that the cluster's sizes are not too different from each other
    # bandwidth now is being chosen by minimizing the cost function
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


def cost_function(data, bandwidth):
    clusters, min, max = rclustering(data, bandwidth_method=bandwidth)
    new_data, error = compress_data(clusters, min, max)
    print("error: ", error)
    w_error = 0
    n_error = []
    for i in range(error.__len__()):
        w_error += np.abs(error[i])
        n_error.append(error[i] / clusters[i].__len__())
    return w_error, n_error, error, clusters.__len__()

# get the optimal bandwidth


def opt_bw(data, min_val=1, max_val=2.5):
    # plot the cost function for bw from 0.1 to 5
    bw = np.arange(0.1, 5, 0.05)
    cost = []
    for i in bw:
        cost.append(cost_function(data, i))
    cost = np.array(cost)
    # convolute error and number of clusters to get a cost function
    y1 = cost[:, 0]
    y1_max = np.max(y1)
    y2 = cost[:, -1]
    y2_max = np.max(y2)
    y1_norm = y1 / y1_max
    y2_norm = y2 / y2_max
    #y = np.convolve(y1, y2, mode='same')
    #y3 = (y1**2) * (y2**3)
    #plt.plot(bw, y)
    #plt.plot(bw, y3)
    # y2 is more important than y1
    answer = []
    for y2_weight in np.arange(min_val, max_val, 0.1):
        #y2_weight = 2
        d = abs(y1_norm - y2_weight*y2_norm)
        #plt.plot(bw, d)
        a = np.where(d == np.min(d))
        #print("optimal bandwidth: ", bw[a])
        answer.append(bw[a])

    # get answer in one array
    answer = np.array(answer)
    answer = answer.reshape(answer.__len__(), 1)

    ans_res = []
    for i in range(answer.__len__()):
        for j in range(answer[i][0].__len__()):
            ans_res.append(answer[i][0][j])
    # sort the answer
    ans_res = np.array(ans_res)
    ans_res = np.sort(ans_res)

    # get optimal bw
    bw_opt = opt_bw_selection(ans_res)

    return bw_opt, ans_res


def opt_bw_selection(ans_res):
    # crop the array into a part where the values distances are maximum 0.1 and can be more at most 1 time
    # get the largest of those values
    # this is the optimal bandwidth
    bw_opt = 0
    _top = 0
    e = 0.001
    for i in range(ans_res.__len__()):
        if i == 0:
            continue
        d = ans_res[i] - ans_res[i-1]
        if d > 0.1 + e:
            if _top == 1:    # if the previous value was the top
                bw_opt = ans_res[i-1]
                break
            _top += 1
    if _top == 1:
        bw_opt = ans_res[-1]
    return bw_opt


def optimal_clustering(data):
    # get bw_opt
    bw_opt, ans_res = opt_bw(data)
    # cluster data
    cluster, min_, max_ = rclustering(data, bandwidth_method=bw_opt)
    # compress data = homogeneous clusters
    clusters, error = compress_data(cluster, min_, max_)
    return clusters, bw_opt, ans_res
