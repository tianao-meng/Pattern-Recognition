import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import copy
import random
import math


def load_dataset(class_num):
    if class_num == 'A':
        text = np.loadtxt("/Users/mengtianao/Documents/SYDE675/a3/classA.csv", dtype=np.str, delimiter=",", encoding='utf-8-sig')
    else:
        text = np.loadtxt("/Users/mengtianao/Documents/SYDE675/a3/classB.csv", dtype=np.str, delimiter=",", encoding='utf-8-sig')

    attribute1 = text[:, 0].astype(np.float)
    attribute2 = text[:, 1].astype(np.float)
    number_sample = len(attribute1)
    attribute = []
    for i in range(number_sample):
        temp = []
        temp.append(attribute1[i])
        temp.append(attribute2[i])

        attribute.append(temp)

    return attribute

def split_data(attribute, label, index_of_test_sample):

    attribute_new = copy.deepcopy(attribute)
    sample_len = len(attribute)
    number_test = 0.1 * sample_len
    start_index = int(index_of_test_sample * number_test - number_test) #0


    end_index = int(start_index + number_test) #10

    test_attribute = []
    test_label = []
    train_attribute = []
    train_label = []

    train_attribute_true = []
    count = 0
    for i in range(len(attribute)):

        if (count >= start_index) and (count < end_index):
            test_attribute.append(attribute_new[i])
            test_label.append(label[i])
            count += 1
            continue

        train_attribute.append(attribute_new[i])
        train_label.append(label[i])
        train_attribute_true.append(attribute[i])
        count += 1

    normal(train_attribute)

    return test_attribute, test_label, train_attribute, train_label, train_attribute_true



def predict_sample(clf, sample_attribute):

    res = clf.predict([sample_attribute])

    return res[0]


def predict(clf, test_attribute):

    predict_res = []

    for i in range(len(test_attribute)):

        predict = predict_sample(clf, test_attribute[i])
        predict_res.append(predict)

    return predict_res

def cross_validation(attribute, label, index_of_test_sample, c, classfier_type):


    test_attribute, test_label, train_attribute, train_label, train_attribute_true = split_data(attribute, label, index_of_test_sample)

    if (classfier_type == 0):

        #print("train_attribute_cross: ", train_attribute)

        clf = svm.SVC(C = c, kernel = 'linear')
        #clf = svm.LinearSVC(C=c)
        clf.fit(train_attribute, train_label)
        predict_res = predict(clf, test_attribute)
        num_wrong = 0
        for i in range(len(predict_res)):
            if (predict_res[i] != test_label[i]):
                num_wrong += 1

        error_rate = num_wrong / len(predict_res)

        return error_rate

    if (classfier_type == 1):
        predict_res = []
        D = []
        for i in range(len(train_attribute)):
            D.append(1 / len(train_attribute))

        res, beta_list = AdaBoostM1(train_attribute_true, train_label, 50, [], [], D)


        for i in test_attribute:

            output = []
            for j in range(len(res)):
                output.append(res[j].predict([i]))

            label_set = set(label)
            label_val = list(label_set)

            count_output = []

            for j in label_val:

                count = 0
                index = 0
                for k in output:

                    if (j == k[0]):

                        if (beta_list[index] == 0):
                            count += math.log2(1 / min(beta_list))
                            index += 1

                            continue

                        count += math.log2(1 / beta_list[index])
                        index += 1

                        continue
                    index += 1

                count_output.append(count)

            output_val_index = count_output.index(max(count_output))
            pre = label_val[output_val_index]

            predict_res.append(pre)

        num_wrong = 0
        #print(len(predict_res))
        for i in range(len(predict_res)):
            if (predict_res[i] != test_label[i]):
                num_wrong += 1

        error_rate = num_wrong / len(predict_res)
        #print(error_rate)
        return error_rate




def dataset_shuffle(attribute, label):

    dataset = copy.deepcopy(attribute)

    for i in range(len(attribute)):
        dataset[i].append(label[i])

    np.random.shuffle(dataset)


    attribute_new = []
    label_new = []
    for i in range(len(dataset)):
        attribute_new.append([dataset[i][0], dataset[i][1]])
        label_new.append(dataset[i][2])

    return attribute_new, label_new



def single_cross_validation(attribute, label, c, classfier_type):

    attribute_random, label_random = dataset_shuffle(attribute, label)
    error_rate = []

    error_rate1 = cross_validation(attribute_random, label_random, 1, c, classfier_type)
    error_rate.append(error_rate1)

    error_rate2 = cross_validation(attribute_random, label_random, 2, c, classfier_type)
    error_rate.append(error_rate2)

    error_rate3 = cross_validation(attribute_random, label_random, 3, c, classfier_type)
    error_rate.append(error_rate3)

    error_rate4 = cross_validation(attribute_random, label_random, 4, c, classfier_type)
    error_rate.append(error_rate4)

    error_rate5 = cross_validation(attribute_random, label_random, 5, c, classfier_type)
    error_rate.append(error_rate5)

    error_rate6 = cross_validation(attribute_random, label_random, 6, c, classfier_type)
    error_rate.append(error_rate6)

    error_rate7 = cross_validation(attribute_random, label_random, 7, c, classfier_type)
    error_rate.append(error_rate7)

    error_rate8 = cross_validation(attribute_random, label_random, 8, c, classfier_type)
    error_rate.append(error_rate8)

    error_rate9 = cross_validation(attribute_random, label_random, 9, c, classfier_type)
    error_rate.append(error_rate9)

    error_rate10 = cross_validation(attribute_random, label_random, 10, c, classfier_type)
    error_rate.append(error_rate10)

    sum_error = 0
    for i in range(len(error_rate)):
        sum_error = sum_error + error_rate[i]
    mean_error_rate = sum_error / len(error_rate)

    sum_var = 0
    for i in range(len(error_rate)):
        sum_var += (error_rate[i] - mean_error_rate) * (error_rate[i] - mean_error_rate)
    var = sum_var / 10
    return mean_error_rate, var

def ten_times_cross_validation(attribute, label, c, classfier_type):
    total_error =[]

    total_var = []
    for i in range(10):
        #print("times: ", i)
        mean_error_rate, var = single_cross_validation(attribute, label, c, classfier_type)
        total_error.append(mean_error_rate)
        total_var.append(var)


    sum_error = 0
    for i in range(len(total_error)):
        sum_error = sum_error + total_error[i]
    mean_error_rate = sum_error / len(total_error)
    #print(mean_error_rate)

    mean_error_rate_var = sum(total_var) / 10
    return mean_error_rate, mean_error_rate_var


def normal(feature):
    sum_x = 0
    sum_y = 0
    count = 0
    sample_max = np.max(feature, axis=0)

    for sample in feature:
        sum_x += sample[0]
        sum_y += sample[1]
        count += 1

    mean_x = sum_x / count
    mean_y = sum_y / count

    for sample in feature:
        sample[0] = (sample[0] - mean_x) / sample_max[0]
        sample[1] = (sample[1] - mean_y) / sample_max[1]
    return feature

def plot_decision_boundary(dataset_attribute, dataset_label, c, attribute, classfier_type):
    if (classfier_type == 0):
        #clf = svm.LinearSVC(C=c)
        clf = svm.SVC(C = c, kernel = 'linear')
        clf.fit(dataset_attribute, dataset_label)

        dataset_w = clf.coef_[0]
        dataset_a = -dataset_w[0] / dataset_w[1]


        dataset_xx = np.linspace(200, 470)


        dataset_yy = dataset_a * dataset_xx - (clf.intercept_[0]) / dataset_w[1]

        margin = 300 / np.sqrt(np.sum(clf.coef_ ** 2))
        yy_down = dataset_yy - np.sqrt(1 + dataset_a ** 2) * margin
        yy_up = dataset_yy + np.sqrt(1 + dataset_a ** 2) * margin
        #print("margin: ", margin)
        if (c == 0.1):

            plt.plot(dataset_xx, dataset_yy, 'k-', label = 'C = 0.1', c = 'r')
            plt.plot(dataset_xx, yy_down, 'k--', c='r')
            plt.plot(dataset_xx, yy_up, 'k--', c='r')
            #print("clf.support_ ", clf.support_vectors_)

        if (c == 1):

            plt.plot(dataset_xx, dataset_yy, 'k-', label = 'C = 1', c = 'y')
            plt.plot(dataset_xx, yy_down, 'k--', c='r')
            plt.plot(dataset_xx, yy_up, 'k--',  c='r')

        if (c == 10):

            plt.plot(dataset_xx, dataset_yy, 'k-', label = 'C = 10', c = 'b')
            plt.plot(dataset_xx, yy_down, 'k--',  c='r')
            plt.plot(dataset_xx, yy_up, 'k--', c='r')

        if (c == 100):

            plt.plot(dataset_xx, dataset_yy, 'k-', label = 'C = 100', c = 'g')
            plt.plot(dataset_xx, yy_down, 'k--', c='r')
            plt.plot(dataset_xx, yy_up, 'k--', c='r')

        #print("error_rate: ", error_rate_cal(dataset_attribute, dataset_label, clf, attribute))
    else:
        attribute = np.array(attribute)
        x_min, x_max = attribute[:, 0].min() - 1, attribute[:, 0].max() + 1
        #print(x_min, x_max)
        y_min, y_max = attribute[:, 1].min() - 1, attribute[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 5),
                             np.arange(y_min, y_max, 5))
        predict_res = []
        D = []
        for i in range(len(attribute)):
            D.append(1 / len(attribute))
        res, beta_list = AdaBoostM1(dataset_attribute, dataset_label, 50,[],[], D)
        #print(res)
        #sum_beta = sum(beta_list)
        #print(beta_list)
        for x in np.c_[xx.ravel(), yy.ravel()]:
            output = []
            for j in range(len(res)):
                output.append(res[j].predict([x]))

            label_set = set(label)
            label_val = list(label_set)

            count_output = []

            for j in label_val:

                count = 0
                index = 0
                for k in output:

                    if (j == k[0]):

                        if (beta_list[index] == 0):
                            count += math.log2(1 / min(beta_list))
                            index += 1

                            continue

                        count += math.log2(1 / beta_list[index])
                        index += 1

                        continue
                    index += 1

                count_output.append(count)

            output_val_index = count_output.index(max(count_output))
            pre = label_val[output_val_index]

            predict_res.append(pre)

        ans = []
        for i in predict_res:
            #print(i)
            if i == 'A':
                ans.append(0)
            else:
                ans.append(1)
        ans = np.array(ans)
        ans = ans.reshape(xx.shape)
        #print(ans)
        plt.contour(xx, yy, ans)

# T number of classfiers
# D sample weights distribution
# res is the classfiers list
# attribute and label are the dataset  #401 include repeated sample
# every time  D = [1/401 ...1/401]
# res is the result list of classfier. length is 50
# we choose c =0.01 for each svm learner
# only choose 100 data of this dataset to train( every time the dataset is 400 samples however may include the repeated sample)

def AdaBoostM1(attribute, label, T, res, beta_list, D):
    #print(D)
    #random.seed(0)
    if T == 0:
        return res, beta_list
    column = len(attribute)
    index_list = range(column)
    train_index = random.sample(index_list, 100)
    train_attribute = []
    train_label = []
    for i in train_index:
        train_attribute.append(attribute[i])
        train_label.append(label[i])
    D_train = []

    #D_new = copy.deepcopy(D)
    for i in train_index:
        D_train.append(D[i])
    D_train = np.array(D_train)
    total_D_train = sum(D_train)
    #print("sum d train ", sum(D_train))
    clf = svm.SVC(C = 0.1, kernel='linear', random_state = 0)
    #clf =svm.LinearSVC(C = 0.1, max_iter=100000)
    clf.fit(train_attribute, train_label, sample_weight = D_train )
    #clf.fit(train_attribute, train_label)
    #clf.fit(train_attribute, train_label, sample_weight=D_train)
    error = 0
    for i in range(len(train_attribute)):
        if ( clf.predict([train_attribute[i]]) != train_label[i]):
            error += D_train[i]
    if (error >= 0.5) :
        return AdaBoostM1(attribute, label, T, res, beta_list,D)
    else:
        # claculate the beta
        res.append(clf)
        #print("error:",error)
        beta = error / (1 - error)
        beta_list.append(beta)
        #print("beta: ", beta)
        #update the sample weights
        predict = clf.predict(train_attribute)
        #print(predict)
        for i in range(len(predict)):
            if (predict[i] == train_label[i]):
                D[train_index[i]] = (D[train_index[i]] * beta)
        #print("sum_d before", sum(D))
        total = sum(D)
        for i in range(len(D)):
            D[i] = D[i] / total
        #print(D)
        return AdaBoostM1(attribute, label, T - 1, res, beta_list, D)




if __name__ == "__main__":

    classA_attribute = load_dataset('A')
    classB_attribute = load_dataset('B')

    classA_attribute_arr = np.array(classA_attribute)

    classB_attribute_arr = np.array(classB_attribute)


    label = []
    attribute = []
    for i in range(len(classA_attribute)):
        attribute.append(classA_attribute[i])
        label.append('A')

    for i in range(len(classB_attribute)):
        attribute.append(classB_attribute[i])
        label.append('B')

    C_value = [0.1, 1, 10, 100]

    label = []
    attribute = []
    for i in range(len(classA_attribute)):
        attribute.append(classA_attribute[i])
        label.append('A')

    for i in range(len(classB_attribute)):
        attribute.append(classB_attribute[i])
        label.append('B')
    #print("attribute: ", attribute)

    attribute_nor = copy.deepcopy(attribute)
    normal(attribute_nor)

    #print("attribute_nor: ",attribute)
    accuarcy = []
    for i in C_value:
        print("c: ", i)
        error_rate, variance_1 = ten_times_cross_validation(attribute, label, i,0)
        accuarcy.append(1 - error_rate)

    print("accuarcy: ", accuarcy)

    plt.figure(1)

    plt.title('visualization')
    plt.scatter(classA_attribute_arr[:,0], classA_attribute_arr[:,1],c = 'r', label = 'classA', alpha=0.6)
    plt.scatter(classB_attribute_arr[:,0], classB_attribute_arr[:,1],c = 'g', label = 'classB', alpha=0.6)
    plt.legend()

    plt.figure(2)

    plt.title('C = 0.1')
    plt.scatter(classA_attribute_arr[:,0], classA_attribute_arr[:,1],c = 'r', label = 'classA', alpha=0.6)
    plt.scatter(classB_attribute_arr[:,0], classB_attribute_arr[:,1],c = 'g', label = 'classB', alpha=0.6)
    plot_decision_boundary(attribute_nor, label, 0.1, attribute,0)
    plt.legend()

    plt.figure(3)
    plt.title('C = 1')
    plt.scatter(classA_attribute_arr[:,0], classA_attribute_arr[:,1],c = 'r', label = 'classA', alpha=0.6)
    plt.scatter(classB_attribute_arr[:,0], classB_attribute_arr[:,1],c = 'g', label = 'classB', alpha=0.6)
    plot_decision_boundary(attribute_nor, label, 1, attribute,0)
    plt.legend()

    plt.figure(4)
    plt.title('C = 10')
    plt.scatter(classA_attribute_arr[:,0], classA_attribute_arr[:,1],c = 'r', label = 'classA', alpha=0.6)
    plt.scatter(classB_attribute_arr[:,0], classB_attribute_arr[:,1],c = 'g', label = 'classB', alpha=0.6)
    plot_decision_boundary(attribute_nor, label, 10, attribute,0)
    plt.legend()

    plt.figure(5)
    plt.title('C = 100')
    plt.scatter(classA_attribute_arr[:,0], classA_attribute_arr[:,1],c = 'r', label = 'classA', alpha=0.6)
    plt.scatter(classB_attribute_arr[:,0], classB_attribute_arr[:,1],c = 'g', label = 'classB', alpha=0.6)
    plot_decision_boundary(attribute_nor, label, 100, attribute,0)
    plt.legend()

    plt.figure(6)

    plt.title('adaboost')
    plot_decision_boundary(attribute, label, 0.1, attribute, 50)
    plt.scatter(classA_attribute_arr[:,0], classA_attribute_arr[:,1],c = 'r', label = 'classA', alpha=0.6)
    plt.scatter(classB_attribute_arr[:,0], classB_attribute_arr[:,1],c = 'g', label = 'classB', alpha=0.6)
    plt.legend()

    mean_error, variance_2 = ten_times_cross_validation(attribute, label, 0, 1)
    print("mean_accuarcy: ",1 - mean_error)
    print("variance: ", variance_2)
    plt.show()












