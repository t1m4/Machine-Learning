import matplotlib.pyplot as plt
import numpy as np

N = 1000
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

def regularisation():
    ind = np.arange(N)
    # перераспределение массива
    np.random.shuffle(ind)
    # Деление на 3 выборки
    ind_train = ind[:int(0.8 * N)]
    ind_valid = ind[int(0.8 * N):int(0.9 * N)]
    ind_test = ind[int(0.9 * N):int(N)]
    x_train = x[ind_train]
    y_train = t[ind_train]
    x_valid = x[ind_valid]
    y_valid = t[ind_valid]
    x_test = x[ind_test]
    y_test = t[ind_test]

    lamb_best = 0
    basic_best = []
    w_best = []
    E_min = 10 ** 10
    lamb = np.array([0.000000001, 0.00000001, 0.0000001, 0.000001,0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000])
    basic = [np.sin, np.cos, np.sqrt, get_func(2),get_func(3),get_func(4),get_func(5),get_func(6),get_func(7),get_func(8),get_func(9)]
    iters_count = 1000


    for i in range(iters_count):
        lamb_cur = get_random_lamb(lamb)
        basic_cur = get_random_basic(basic)

        w_cur=get_w(basic_cur, lamb_cur,x_train, y_train)
        E_cur=get_error(w_cur, basic_cur, x_valid, y_valid)

        if E_cur < E_min:
            E_min = E_cur
            lamb_best = lamb_cur
            basic_best = basic_cur
            w_best = w_cur

    for j in range(len(basic_best)):
        print("The best basic function - ", basic_best[j].__name__)
    print("The best lambda - ", lamb_best)
    #print("The best w - ", w_best)

    E_test = get_error(w_best, basic_best, x_test, y_test)
    print("This is error -----------------", E_test)

    mat_d = get_mat_d(x, basic_best)
    y = np.dot(mat_d, w_best)
    fig = plt.figure()
    plt.plot(x,z, "r",ms=3)
    plt.plot(x,t, "b.",ms=2)
    plt.plot(x,y, "g",ms=3)
    plt.show()

def get_func(index):
    a = lambda x: x**index
    a.__name__ = "Polynom degree " + str(index)
    return a

def get_random_lamb(lamb):
    index = np.random.randint(1, lamb.size)
    return lamb[index]

def get_random_basic(basic):
    return np.random.choice(basic, np.random.randint(1, len(basic)), replace=False)

def get_mat_d(x_cur, basic_cur):
    M = len(basic_cur) + 1
    N = len(x_cur)
    mat_d = np.zeros((N, M))
    mat_d[:, 0] = 1

    for i in range(1, M):
        mat_d[:, i] = basic_cur[i - 1](x_cur)

    return mat_d

def get_w(basic, lamb,x_train,  t_cur):
    M = len(basic) + 1
    I = np.zeros((M, M))
    np.fill_diagonal(I,1)

    mat_d = get_mat_d(x_train, basic)
    w = np.dot(np.dot(np.linalg.inv(np.dot(mat_d.T, mat_d) + lamb * I) , mat_d.T), t_cur)
    return w

def get_error(w_cur,  basic_cur, x_cur, t_cur):
    mat_d_cur = get_mat_d(x_cur, basic_cur)
    y = np.dot(mat_d_cur, w_cur)

    E = 0
    for j in range(len(t_cur)):
        E += (y[j] - t_cur[j]) ** 2
    E = E * 0.5
    return E


regularisation()
