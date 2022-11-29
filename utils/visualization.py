import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils import geometry as geo
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def plot_data(data, size=10, limits=[-10, 10, -10, 10]):
    fig, ax = plt.subplots(figsize=(size, size))
    patches = []
    num_polygons = data.__len__()

    for i in range(num_polygons):
        square = geo.create_square2(
            data[i][0], data[i][1], data[i][2], data[i][3])
        polygon = Polygon(square, True)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

    colors = 100*np.random.rand(len(patches))
    p.set_array(np.array(colors))

    ax.add_collection(p)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    plt.show()
