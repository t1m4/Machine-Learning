import matplotlib.pyplot as plt
import numpy as np

N=1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

def linearRegression(M, n):
    mat_d = np.zeros((N,M))
    #Создаем массив Ф
    for i in range(M):
        mat_d[:,i] = x ** i

    w = np.dot(np.dot(np.linalg.inv(np.dot(mat_d.T, mat_d)), mat_d.T), t)
    #решение регрессии
    y = np.dot(mat_d, w)

    ax1 = fig.add_subplot(1,3,n)
    ax1.plot(x,z)#функция z(x)
    ax1.plot(x,t,"r.",ms=1)#функия t(x)
    ax1.plot(x,y)#решение регрессии для M=1,8,100

fig = plt.figure()
linearRegression(1,1)
linearRegression(8,2)
linearRegression(100,3)
plt.show()

def linear(M):
    mat_d = np.zeros((N, M))
    # Создаем массив Ф
    for i in range(M):
        mat_d[:, i] = x ** i

    w = np.dot(np.dot(np.linalg.inv(np.dot(mat_d.T, mat_d)), mat_d.T), t)
    #решение регрессии
    y = np.dot(mat_d, w)
    return y
#функция для графика зависимости ошибки
def liniarError():
    mat_k = [i for i in range(1,101)]
    for k in range(1,101):
        y = linear(k)
        mat_E = np.zeros(100)
        E = 0
        for j in range(N):
            E+=(y[j]-t[j])**2
        E=E*0.5
        mat_E[k-1]=E;
        plt.plot(mat_k,mat_E)
    plt.show()
#liniarError()