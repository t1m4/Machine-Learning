from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import math


def get_params(data, target):

    list_left_target = []
    list_right_target = []
    index_best = 0
    inf_best = 0
    t_best = 0
    for j in range(data.shape[1]):  # =j
        #components = data[:, j]
        for t in range(17):

            for i in range(len(data)):
                if data[i, j] < t:
                    list_left_target.append(target[i])
                else:
                    list_right_target.append(target[i])

            left_target = np.array(list_left_target)
            right_target = np.array(list_right_target)
            list_right_target.clear()
            list_left_target.clear()
            inf_temp = information_gain(left_target, right_target, target)

            if inf_temp >= inf_best:
                inf_best = inf_temp
                t_best = t
                index_best = j  # = j

    return t_best, index_best


def information_gain(left_t, right_t, target):
    num = target.shape[0]
    num_left = left_t.shape[0]  # 0 or 1 ????
    num_right = right_t.shape[0]
    inf = (entropy_calculation(target) - ((num_left/num)*entropy_calculation(left_t) + (num_right/num)*entropy_calculation(right_t)))
    return inf


def entropy_calculation(target):
    num = target.shape[0]
    character, num_i = np.unique(target, return_counts=True)

    arr_num = np.zeros(10)
    for i in range(character.shape[0]):
        arr_num[character[i]] = num_i[i]

    h = 0
    for k in range(10):
        if arr_num[k] != 0:
            h -= ((arr_num[k] / num) * math.log(arr_num[k] / num))

    return h


def split_data(data, target, threshold, index):

    list_left_target = []
    list_right_target = []
    list_left_data = []
    list_right_data = []

    components = data[:, index]
    for i, x in enumerate(components):
        if x <= threshold:
            list_left_target.append(target[i])
            list_left_data.append(data[i])
        else:
            list_right_target.append(target[i])
            list_right_data.append(data[i])

    left_target = np.array(list_left_target)
    left_data = np.array(list_left_data)
    right_target = np.array(list_right_target)
    right_data = np.array(list_right_data)

    return left_data, right_data, left_target, right_target


def create_split_node(threshold, index):

    node = {
        'coord_ind': index,
        'threshold': threshold,
        'is_terminal': False,
        'left': None,
        'right': None
    }

    return node


def stop_criteria(depth, target):
    num = target.shape[0]
    h = entropy_calculation(target)
    if depth > 0 and num > 50 and h > 0.4:
        return 0
    else:
        return 1


def create_terminal_node(target):

    assure_list = []
    num = target.shape[0]
    character, num_i = np.unique(target, return_counts=True)

    arr_num = np.zeros(10)
    for i in range(character.shape[0]):
        arr_num[character[i]] = num_i[i]

    # for k in range(10):
    #     assure_list.append((arr_num[k] / num))


    if num > 0:
        for k in range(10):
            assure_list.append((arr_num[k] / num))
    else:
        for k in range(10):
            assure_list.append(arr_num[k])

    # t_vector = np.array(assure_list)

    node = {'is_terminal': True, 'num': num, 'vector': assure_list}

    return node


def create_tree(data, target, depth):

    if not stop_criteria(depth, target):
        depth -= 1
        threshold, index = get_params(data, target)
        left_data, right_data, left_target, right_target = split_data(data, target, threshold, index)
        node = create_split_node(threshold, index)
        node["left"] = create_tree(left_data, left_target, depth)
        node["right"] = create_tree(right_data, right_target, depth)
    else:
        node = create_terminal_node(target)

    return node


def tree_solution(data, node):

    if not node["is_terminal"]:
        if data[node["coord_ind"]] < node["threshold"]:
            return tree_solution(data, node["left"])
        else:
            return tree_solution(data, node["right"])
    else:
        return node['vector']


def metric_calculation(data, target, node):

    confusion_matrix = np.zeros((10, 10))
    y_matrix = np.zeros((data.shape[0], 10))

    for i in range(data.shape[0]):
        y_matrix[i] = np.array(tree_solution(data[i], node))

    for k in range(y_matrix.shape[0]):
        confusion_matrix[target[k]][np.argmax(y_matrix[k])] += 1

    correct_list = []
    incorrect_list = []
    for i in range(data.shape[0]):
        if target[i] == np.argmax(y_matrix[i]):
            correct_list.append(np.amax(y_matrix[i]))
        else:
            incorrect_list.append(np.amax(y_matrix[i]))

    correct_arr = np.array(correct_list)
    incorrect_arr = np.array(incorrect_list)

    temp = 0  # accuracy

    for i in range(confusion_matrix.shape[0]):
        temp += confusion_matrix[i][i]

    return confusion_matrix, temp / y_matrix.shape[0], correct_arr, incorrect_arr


def histogram_creation(corr_train, incorr_train, corr_test, incorr_test):

    fig = plt.figure()
    ax_1 = fig.add_subplot(2, 2, 1)
    ax_2 = fig.add_subplot(2, 2, 2)
    ax_3 = fig.add_subplot(2, 2, 3)
    ax_4 = fig.add_subplot(2, 2, 4)

    ax_1.hist(corr_train)
    ax_1.set_title("Правильность Train")
    ax_2.hist(incorr_train)
    ax_2.set_title("Неправильность Train")
    ax_3.hist(corr_test, color='red')
    ax_3.set_title("Правильность Test")
    ax_4.hist(incorr_test, color='red')
    ax_4.set_title("Неправильность Test")

    plt.show()


digits = load_digits()

N = digits.data.shape[0]

ind = np.arange(N)
np.random.shuffle(ind)

ind_train = ind[:np.int32(0.8*len(ind))]
ind_test = ind[int(0.8 * digits.data.shape[0]):]
# ind_test = ind[np.int32(0.8*len(ind)):np.int32(len(ind))]

data_train = digits.data[ind_train]
target_train = digits.target[ind_train]

data_test = digits.data[ind_test]
target_test = digits.target[ind_test]

tree = create_tree(data_train, target_train, 15)

confusion_matrix_train, accuracy_train, correct_train, incorrect_train = metric_calculation(data_train, target_train, tree)
confusion_matrix_test, accuracy_test, correct_test, incorrect_test = metric_calculation(data_test, target_test, tree)

print(tree)
print("Confusion matrix for train:\n {0} \n accuracy for train: {1}\n" .format(confusion_matrix_train, accuracy_train))
print("Confusion matrix for test:\n {0} \n accuracy for test: {1}\n" .format(confusion_matrix_test, accuracy_test))

histogram_creation(correct_train, incorrect_train, correct_test, incorrect_test)