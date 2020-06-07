from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import time
start_time = time.time()
centers = [[-1, -1], [0, 1], [1, -1]]
X, _ = make_blobs(n_samples=3000, centers=centers, cluster_std=0.5)

def get_distance(number, center):#return list of distance to centers
    d_list = []
    for i in range(len(center)):
        d_list.append(np.sqrt((number[0]-center[i][0])**2 + (number[1]-center[i][1])**2))
    return (np.argmin(d_list), np.min(d_list))

def get_error(distance):#return sum our vector
    return np.sum(distance)

def get_D(error):#return list_D
    list_D = np.zeros(len(error)-1)
    for i in range(1, len(error) - 1):
        list_D[i] = np.abs(error[i] - error[i+1]) / np.abs(error[i - 1] - error[i])
    return list_D[1:]


def show_claster(data, claster, center):
    colors = ['r', 'g', 'b', 'brown', "coral", "gold", "indigo", "maroon", "olive", 'black']
    list_of_color = {}
    for i in range(len(colors)):
        list_of_color[i] = colors[i]
    for i in range(len(data)):
        color = list_of_color[claster[i]]
        plt.plot(data[i][0], data[i][1], ".", ms=3, color=color)
    for i in range(len(center)):
        print(center[i])
        plt.plot(center[i][0], center[i][1], ".", ms=7, color="black")
    plt.show()

def show(claster_error, list_D):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot([i for i in range(1, len(claster_error)+1)], claster_error)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot([i for i in range(2, len(list_D)+2)], list_D)  # функция z(x)
    plt.show()

def claster(data):
    k = 11
    clasters_index = np.zeros((k, len(data)))
    clasters_index.fill(-1)
    clasters_distanse = np.zeros((k, len(data)))
    clasters_error = np.zeros(k-1)
    clasters_centers = []
    for i in range(1, k):
        print("Claster - ", i)
        center_ind = np.random.randint(0, 3000, i)#INIT centers
        center = data[center_ind]
        #clasters_centers[i] += center
        clasters_centers.append(center)
        for j in range(len(data)):
            cluster_index, distanse = get_distance(data[j], center)
            clasters_index[i][j] = cluster_index
            clasters_distanse[i][j] = distanse
            size = len([l for l in clasters_index[i] if l == cluster_index])#находим количество элементов в этом кластере
            center[cluster_index] = (center[cluster_index]*size + data[j])/(size+1)#+1 так как есть еще центроид

        E = get_error(clasters_distanse[i])
        clasters_error[i-1] = E

    print("Clasters error - ", clasters_error)
    D = get_D(clasters_error)
    print("D - ", D,)
    print("Now you see clasters_error and D....")
    show(clasters_error, D)
    print("Now you see the best claster....")
    show_claster(data, clasters_index[np.argmin(D)+2], clasters_centers[np.argmin(D)+1])#+1 так как D начинается с D(1), +2 так как claster_index начинается с 1

claster(X)

print("This is time - ", time.time() - start_time)
