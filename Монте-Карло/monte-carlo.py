import matplotlib.pyplot as plt
import numpy as np

a = 1
r = 0.5
N = 2000

#1 и 3 task
def calcPI():
    plt.figure()
    #рисует квадрат и круг
    x = [0, 0, 1, 1, 0]
    y = [0, 1, 1, 0, 0]
    plt.plot(x, y)
    circle = np.linspace(0, 2 * np.pi, 100)
    x = 0.5 + r * np.cos(circle)
    y = 0.5 + r * np.sin(circle)
    plt.plot(x,y)

    Ncircle = 0
    points = np.random.rand(N, 2)  # равномерное распределение
    for i in range(N):
        if pow(points[i,0] - a/2,2) + pow(points[i,1] - a/2,2) <= pow(r,2):
            Ncircle+=1
            plt.plot(points[i, 0], points[i, 1], "r.:")
        else:
            plt.plot(points[i, 0], points[i, 1], "b.:")

    PI = Ncircle * pow(a,2) / N / pow(r,2)
    plt.show()
    return PI

print(calcPI())

#2 задание
def paintPI_N(startN, endN):

    points = np.random.rand(endN, 2)  # равномерное распределение
    #узнать количество итераций
    for j in range(startN, endN,10):
        Ncircle = 0
        #подсчет числа PI
        for i in range(j):
            if pow(points[i, 0] - a / 2, 2) + pow(points[i, 1] - a / 2, 2) <= pow(r, 2):
                PI = Ncircle * pow(a, 2) / j / pow(r, 2)
                Ncircle += 1

        plt.plot(j,PI, "r.:")#цвет, форма, тип линий
    plt.show()

paintPI_N(50,10000)
