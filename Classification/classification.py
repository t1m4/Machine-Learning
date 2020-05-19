import matplotlib.pyplot as plt
import numpy as np

N =1000
mu_1 = 193
mu_0 = 180
sigma_1 = 7
sigma_0 = 6
c_1 =  np.random.randn(N) * sigma_1 + mu_1
c_0 =  np.random.randn(N) * sigma_0 + mu_0
c_1_array = np.zeros((1000))
c_1_array.fill(1)
c_0_array = np.zeros((1000))


def get_accuracy(TP, TN):
    return  (TP + TN) / (2 * N)

def get_precision(TP, FP):
    if TP + FP == 0:
        return 1
    else:
        return TP / (TP + FP)

def get_recall(TP, FN):
    return TP/(TP+FN)

def get_score(Precision, Recall):
    return 2*Precision*Recall/(Precision+Recall)

def get_alpha(FP, TN):
    return FP/(TN+FP)

def get_beta(FN, TP):
    return FN/(TP+FN)

def get_class(t):
    for i in range(N):
        if c_1[i]> t:
            c_1_array[i] = 1
        else:
            c_1_array[i] = 0
        if c_0[i]> t:
            c_0_array[i] = 1
        else:
            c_0_array[i] = 0

get_class(183)
TP = 0#он был С1 и предсказали С1
TN = 0#он был С0 и предсказали С0
FP = 0#он был С0 но предсказали С1
FN = 0#он был С1 но предсказали С0
for i in range(N):
    if c_1_array[i] == 1:
        TP+=1
    else:
        FN+=1
    if c_0_array[i] == 0:
        TN+=1
    else:
        FP+=1
print('TP - ', TP, '\nTN - ', TN, '\nFP - ', FP, '\nFN - ', FN)
print("Accuracy", get_accuracy(TP, TN))
Precision = get_precision(TP, FP)
print("Precision - ", Precision)
Recall = get_recall(TP, FN)
print("Recall - ", Recall)
print("F1 score - ", get_score(Precision, Recall))
print("aplha - ",get_alpha(FP, TN))
print("beta - ", get_beta(FN, TP))

def get_roc():
    coordinate = [0,0]
    alpha_list = np.zeros(300)
    recall_list =np.zeros(300)
    best_accuracy = 0
    best_j = 0
    confusion = [[0,0],[0, 0]]
    for j in range(300):
        get_class(j)
        TP = 0  # он был С1 и предсказали С1
        TN = 0  # он был С0 и предсказали С0
        FP = 0  # он был С0 но предсказали С1
        FN = 0  # он был С1 но предсказали С0
        for i in range(N):
            if c_1_array[i] == 1:
                TP += 1
            else:
                FN += 1
            if c_0_array[i] == 0:
                TN += 1
            else:
                FP += 1
        accuracy = get_accuracy(TP, TN)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_j = j
            confusion[0][0] = TN
            confusion[1][1] = TP
            confusion[1][0] = FN
            confusion[0][1] = FP
        alpha_list[j] = FP / (TN + FP)
        recall_list[j] = TP / (TP + FN)

    print("Best t - ", best_j)
    print("Best accuracy - ", best_accuracy)
    TP = confusion[1][1]
    TN = confusion[0][0]
    FP = confusion[0][1]
    FN = confusion[1][0]
    print('Best TP - ', TP, '\nBest TN - ', TN, '\nBest FP - ', FP, '\nBest FN - ', FN)
    Precision = get_precision(TP, FP)
    print("Best Precision - ", Precision)
    Recall = get_recall(TP, FN)
    print("Best Recall - ", Recall)
    print("Best F1 score - ", get_score(Precision, Recall))
    print("Best aplha - ", get_alpha(FP, TN))
    print("Best beta - ", get_beta(FN, TP))


    coordinate[0] = alpha_list#x
    coordinate[1] = recall_list#y
    plt.plot(alpha_list, recall_list)
    return coordinate

def get_auc():
    sum = 0
    roc = get_roc()
    x_0 = 1
    for i in range(299):
        a = x_0 - roc[0][i]
        b = x_0 - roc[0][i+1]
        h = roc[1][i] - roc[1][i+1]
        sum+= (a+b)/2*h
    print("AUC - ", sum)
    plt.show()


get_auc()