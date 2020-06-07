from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import time
time_start = time.time()

digits = load_digits()

'''plt.figure()
plt.imshow(digits.images[1], cmap='gray') # img - изображение представленное в виде массива 8х8
plt.show()'''
data = digits.data
target = digits.target
class Regression:
    def __init__(self, data, target):
        self.K = len(data[0])
        self.data = self.standartization(data)
        self.target = self.get_one_hot_encoding(target)
        self.data_train, self.target_train, self.data_valid, self.target_valid = self.set_division(self.data, self.target)

    def standartization(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        for i in range(len(data)):
            for j in range(len(data[0])):
                if sigma[j]!=0:
                    data[i][j] = (data[i][j] - mu[j]) / sigma[j]
        return data
    def get_one_hot_encoding(self, target):
        result_target = np.zeros((len(target), 10))
        for i in range(len(target)):
            result_target[i][target[i]] = 1
        return result_target
    def set_division(self, data, target):#
        # Деление на 2 выборки
        N = len(data)
        ind = np.arange(N)
        # перераспределение массива
        np.random.shuffle(ind)
        ind_train = ind[:int(0.8 * N)]
        ind_valid = ind[int(0.8 * N):int(N)]
        data_train = data[ind_train]
        target_train = target[ind_train]
        data_valid = data[ind_valid]
        target_valid = target[ind_valid]
        return (data_train, target_train, data_valid, target_valid)


    def softmax(self, a):#return vector (size = (10 ,1))
        a -=max(a)
        return np.exp(a) / sum(np.exp(a))
    def get_Y(self, data, w, b):
        result = np.zeros((len(data), 10))
        for i in range(len(data)):
            res = w @ data[i] + b
            result[i] = self.softmax(res)
        return result


    def gradient(self, data, target, y):#return vector nabla_W and nabla_b
        U = np.zeros(len(data))
        U.fill(1)
        nabla_W = (y - target).T @ data  # (len(train) * 10 - len(train) * 10).T * (len(train), 64) = (10, 64)
        nabla_b = (y - target).T @ U
        return (nabla_W, nabla_b)

    def get_error(self, target, y):#return error
        result = 0
        for i in range(len(y)):
            for j in range(10):
                result += target[i][j]*np.log(y[i][j])
        return -result


    def get_accuracy(self, y, target):#return accuracy
        TP = self.get_metric(y, target)
        return TP / len(y)

    def get_metric(self, y, target):
        TP = 0
        for i in range(len(y)):
            if np.argmax(y[i]) == np.argmax(target[i]):
                TP += 1
        return TP

    def get_graphic(self, error_valid_list, error_train_list):
        print("Valid error - ", error_valid_list[-1])
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot([i for i in range(len(error_train_list))], error_train_list)  # функция z(x)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot([i for i in range(len(error_valid_list))], error_valid_list)  # функция z(x)
        plt.show()

    def start_regresion(self):
        sigma = 0.01
        gamma = 0.001

        print("[+] Start softmax regresion")
        w = sigma * np.random.randn(10, 64)
        b = sigma * np.random.randn(10)
        error_valid_list = []
        error_train_list = []
        j = 0
        while j < 101:
            y_train = self.get_Y(self.data_train, w, b)
            y_valid = self.get_Y(self.data_valid, w, b)

            W = self.gradient(self.data_train, self.target_train, y_train)
            w = w - gamma * W[0]
            b = b - gamma * W[1]

            E_train = self.get_error(self.target_train, y_train)
            E_valid = self.get_error(self.target_valid, y_valid)
            error_valid_list.append(E_valid)
            error_train_list.append(E_train)
            if j%10 == 0:
                print("This is error train set - ", j, E_train,  " and Accuracy - ",self.get_accuracy(y_train, self.target_train))
                print("This is error valid set - ", j, E_valid,  " and Accuracy - ",self.get_accuracy(y_valid, self.target_valid), "\n\n")

            if j!=0:
                if error_valid_list[j-1] - error_valid_list[j] < 0.05:
                    break
            j += 1

        #print("Before graph - ", time.time() - time_start)
        self.get_graphic(error_valid_list, error_train_list)


element = Regression(data, target)
element.start_regresion()


time_end = time.time()
print("\n\nIt is time - ",time_end - time_start)