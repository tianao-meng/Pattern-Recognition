import numpy as np
import copy
#atribute_num means node
#atribute_values is the list of the value of the feature, eg. [windy, rainy..]
#children is the list of subtree, each index respond to the relevant value of the atribute
#leafvalue is a bool, indicate whether the node is leaf

class DC:

    def __init__(self, atribute_num = -1, atribute_value=[], Children=[], LeafValue=None):
        #if atribute_value is None:
            #atribute_value = []
        self.atribute_num = atribute_num
        self.atribute_value = atribute_value
        self.Children = Children
        self.Leaf_value = LeafValue

class DC_Con:

    def __init__(self, atribute_num=-1, atribute_value=[], Children=[], LeafValue=None, Threshold=-1):
        # if atribute_value is None:
        # atribute_value = []
        self.atribute_num = atribute_num

        self.Children = Children
        self.Leaf_value = LeafValue
        self.threshold = Threshold

def get_attribute_value(attribute):
    attribute_value = set()
    for i in attribute:
        attribute_value.add(i)
    res = list(attribute_value)
    #res = ['x', 'o', 'b']
    return res
"""
def get_res(decision):
    res =['positive','negative']
    return res
"""
def choose_best_attribute(dataset, available_attribute):

    #best_information_gain = 0
    #inter = float('inf')
    decision = dataset['decision']

    attribute_infomation_gain_dic = {}
    for key in dataset:

        if (key == 'decision'):
            break

        if (key not in available_attribute):
            continue

        attribute = dataset[key]
        information_gain_attribute = information_gain(attribute, decision)
        attribute_infomation_gain_dic[key] = information_gain_attribute
        # best_information_gain = inter
    #print("attribute_infomation_gain_dic: ", attribute_infomation_gain_dic)
    best_attribute = max(attribute_infomation_gain_dic, key = attribute_infomation_gain_dic.get)
    #print("best_attribute: ", best_attribute)
    #print(best_attribute)
    return best_attribute

#return a dic
def get_attribute_value_subset(dataset, attribute_key, attribute_value):

    index_list =[]
    attribute_list = dataset[attribute_key]
    index = 0
    #get index for attribute_value
    for i in attribute_list:
        if (i == attribute_value):
            index_list.append(index)
        index += 1

    sub_dataset ={}

    for key in dataset:
        sub_attribute_value =[]
        for i in index_list:
            sub_attribute_value.append(dataset[key][i])

        sub_dataset[key] = sub_attribute_value

    return sub_dataset


def check_all_same(decision):
    #print("decision: ", decision)
    inter = decision[0]
    for i in decision:
        if (i != inter):
            return False

    return True

def deep_copy_list(list):
    new = []
    for i in list:
        new.append(i)
    return new

def id3(dataset, chosen_list):

    available_attribute =[]
    for key in dataset:
        if key not in chosen_list and key != 'decision':
            available_attribute.append(key)


    #get the key list
    total_list = []
    attribute_list = []
    feature_num = 0
    attribute_value_dic = {} #include decision
    for key in dataset:
        total_list.append(key)

        if (key != 'decision'):
            feature_num += 1
            attribute_list.append(key)
        #include decision
        attribute_value_dic[key] = get_attribute_value(dataset[key])
        """
        if(key == 'decision'):
            attribute_value_dic[key] = get_res(dataset[key])
        """
    #print("chosen list", chosen_list)
    if len(attribute_list) == len(chosen_list) or check_all_same(dataset['decision']) == True:
        count_max = float("-inf")

        max_index = 0
        for i in (attribute_value_dic['decision']):
            count = 0
            for j in (dataset['decision']):
                if j == i:
                    count += 1
            if (count > count_max):
                max_index = attribute_value_dic['decision'].index(i)
        #print(attribute_value_dic['decision'][max_index])
        if chosen_list != []:
            del chosen_list[-1]
        leaf_node = DC(atribute_num = attribute_value_dic['decision'][max_index], atribute_value=[], Children=[], LeafValue=True)
        return leaf_node

    root = DC(atribute_num = -1, atribute_value=[], Children=[], LeafValue=None)
    best_attribute = choose_best_attribute(dataset, available_attribute)
    root.atribute_num = best_attribute
    root.atribute_value = attribute_value_dic[best_attribute]
    #print(root.atribute_value)
    chosen_list.append(best_attribute)

    #chosen_list_copy = deep_copy_list(chosen_list)
    #print("attribute_value_dic[attribute 4] ", attribute_value_dic['attribute_4'])
    for i in attribute_value_dic[best_attribute]:

        #dataset_copy = dataset.copy()
        best_attribute_subset = get_attribute_value_subset(dataset, best_attribute, i)
        #chosen_list = chosen_list_copy

        #print("beat_attribute_sbuset: ", best_attribute_subset)
        #print("chosen list: ",chosen_list)
        if(best_attribute_subset['decision'] == []):
            leaf_node = DC(atribute_num='positive', atribute_value=[], Children=[], LeafValue=True)
            if chosen_list != []:
                del chosen_list[-1]
            return leaf_node
        childre_node = id3(best_attribute_subset, chosen_list)
        root.Children.append(childre_node)

    if chosen_list != []:
        del chosen_list[-1]
    return root
"""
def get_each_sample(dataset):
    sample_num = len(dataset['decision'])
    each_sample = {}
    res = []
    for key in dataset:
        dataset[key]
"""



def decision_classification(decision):

    decision_value = set()

    for i in decision:
        decision_value.add(i)

    decision_value_num = len(decision_value)

    decision_classfication =[]
    for i in range(decision_value_num):
        decision_classfication.append([])

    for i in decision:
        for j in range(decision_value_num):
            if i in decision_classfication[j]:
                decision_classfication[j].append(i)
                break


            if i not in decision_classfication[j] and len(decision_classfication[j]) == 0:
                decision_classfication[j].append(i)
                break

    return decision_classfication

def entropy_cal(decision):
    entropy = 0

    sample_num = len(decision)
    decision_seperate = decision_classification(decision)

    decision_value_num = len(decision_seperate)


    for i in range(decision_value_num):
        probability = len(decision_seperate[i]) / sample_num
        #print(probability)
        entropy += (-probability * np.log2(probability))
    return entropy

#def conditional_probability(attribute_value, decision_value):

def conditional_entropy(attribute, decision):
    #get attribute value, transferred to a list
    attribute_value = set()
    for i in attribute:
        attribute_value.add(i)
    # attribute_value_list =["rain", "windy"]
    attribute_value_list = list(attribute_value)

    #attribute_value_num = 2
    attribute_value_num = len(attribute_value)

    #attribute_value_dic = {rain: [play, play,..], windy: [notplay, play ...]
    attribute_value_dic = {}
    for i in range(attribute_value_num):
        attribute_value_dic[attribute_value_list[i]] =[]
    #attribute = [rain, windy, windy, rain...] number = sample_num
    index = 0
    for i in attribute:
        attribute_value_dic[i].append(decision[index])
        index += 1


    #get entropy
    res = 0
    sample_num = len(attribute)
    for i in attribute_value_list:
        count = 0

        for j in attribute:
            if (j == i):
                count += 1

        p_i = count / sample_num
        con_entropy = entropy_cal(attribute_value_dic[i])
        res += p_i * con_entropy

    return res

def information_gain(attribute, decision):
    total_entroy = entropy_cal(decision)
    con_entropy = conditional_entropy(attribute, decision)
    res = total_entroy - con_entropy
    return res


#assume we have 10 fold, index_of_test_sample is the ith fold as test data
#total sample number = 100
#number_test = 10
#1 - 0 - 9
#2 - 10 -19
#3 - 20 -39
def split_data(dataset, index_of_test_sample):

    sample_len = len(dataset['decision'])
    number_test = 0.1 * sample_len
    start_index = int(index_of_test_sample * number_test - number_test) #0
    #print("start_index: ", start_index)

    end_index = int(start_index + number_test) #10
    #print("end_index: ", end_index)
    test_dataset = {}
    train_dataset = {}
    for key in dataset:

        train_attribute =[]
        test_attribute = dataset[key][start_index : end_index]
        test_dataset[key] = test_attribute
        for i in range(len(dataset[key])):
            if (start_index <= i) and (i < end_index):
                #print("test i: ", i)
                continue
            train_attribute.append(dataset[key][i])
        train_dataset[key] = train_attribute

    return test_dataset, train_dataset


def predict_sample(dataset, tree, sample):

    if tree.Leaf_value == True:
        res = tree.atribute_num
        #print("res:", res)
        return res

    attribute = tree.atribute_num
    #print(attribute)
    branch = sample[attribute]
    #print(branch)

    if branch in tree.atribute_value:

        attribute_value_index = tree.atribute_value.index(branch)
        sub_tree = tree.Children[attribute_value_index]
        res = predict_sample(dataset, sub_tree, sample)
        return res

    if branch not in tree.atribute_value:
        #print("branch not in tree.atribute_value")
        count_pos = 0
        count_neg = 0
        decision = dataset['decision']
        for i in decision:
            if i == 'positive':
                count_pos += 1
            else:
                count_neg += 1
        if(count_pos > count_neg):
            res ='positive'

        else:
            res ='negative'

        return res


"""
    if(branch <= tree.threshold):
        sub_tree = tree.Children[0]
    else:
        sub_tree = tree.Children[1]
"""





def predict(tree, test_dataset):

    test_sample ={}
    predict_res = []
    actual_res = []
    for i in range(len(test_dataset['decision'])):
        for key in test_dataset:
            if (key != 'decision'):
                test_sample[key] = test_dataset[key][i]
            else:
                actual_res.append(test_dataset[key][i])
        #print("test sample: ",test_sample)
        predict = predict_sample(test_dataset, tree, test_sample)
        predict_res.append(predict)
    #print("predict res: ", predict_res)
    #print("actural res: ", actual_res)
    return predict_res, actual_res

def cross_validation(dataset, index_of_test_sample):


    test1_dataset, train1_dataset = split_data(dataset, index_of_test_sample)
    #print("train dataset: ", train1_dataset)
    #print("test dataset: ", test1_dataset)
    tree = id3(train1_dataset,[])
    predict_res, actual_res = predict(tree, test1_dataset)

    num_wrong = 0
    for i in range(len(predict_res)):
        if(predict_res[i] != actual_res[i]):
            num_wrong += 1

    confusion_matrix = draw_confusion_matrix(predict_res, actual_res)
    error_rate = num_wrong / len(predict_res)

    return error_rate, confusion_matrix

def draw_confusion_matrix(predict_res, actual_res):
    confusion_matrix = np.array([[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0]])
    for i in range(len(predict_res)):
        if (predict_res[i] == actual_res[i]) and (predict_res[i]  == 'negative'):
            confusion_matrix[0][0] += 1

        if (predict_res[i] == actual_res[i]) and (predict_res[i]  == 'positive'):
            confusion_matrix[1][1] += 1

        if (predict_res[i] != actual_res[i]) and (predict_res[i]  == 'negative') and (actual_res[i]  == 'positive'):
            confusion_matrix[0][1] += 1

        if (predict_res[i] != actual_res[i]) and (predict_res[i]  == 'positive') and (actual_res[i]  == 'negative'):
            confusion_matrix[1][0] += 1

    confusion_matrix[0][2] = confusion_matrix[0][0] + confusion_matrix[0][1]
    confusion_matrix[1][2] = confusion_matrix[1][0] + confusion_matrix[1][1]


    confusion_matrix[2][0] = confusion_matrix[0][0] + confusion_matrix[1][0]
    confusion_matrix[2][1] = confusion_matrix[0][1] + confusion_matrix[1][1]


    confusion_matrix[2][2] = confusion_matrix[0][2] + confusion_matrix[1][2]

    return confusion_matrix
def dataset_shuffle(dataset):

    dataset_array = []
    for key in dataset:
        dataset_array.append(dataset[key])
    dataset_array_np = np.array(dataset_array).T
    np.random.shuffle(dataset_array_np)

    shuffle_dataset = {}
    count = 0
    for key in dataset:
        a = dataset_array_np.T.tolist()
        shuffle_dataset[key] = a[count]
        count += 1
    return shuffle_dataset

def single_cross_validation(dataset):

    dataset_random = dataset_shuffle(dataset)


    error_rate = []
    confusion_matrix_list = []

    error_rate1, confusion_matrix1 = cross_validation(dataset_random, 1)
    error_rate.append(error_rate1)
    confusion_matrix_list.append(confusion_matrix1)

    #print("error_rate1: ", error_rate1)
    error_rate2, confusion_matrix2 = cross_validation(dataset_random, 2)
    #print("error_rate2: ", error_rate2)
    error_rate.append(error_rate2)
    confusion_matrix_list.append(confusion_matrix2)

    error_rate3, confusion_matrix3 = cross_validation(dataset_random, 3)
    #print("error_rate3: ", error_rate3)
    error_rate.append(error_rate3)
    confusion_matrix_list.append(confusion_matrix3)

    error_rate4, confusion_matrix4 = cross_validation(dataset_random, 4)
    #print("error_rate4: ", error_rate4)
    error_rate.append(error_rate4)
    confusion_matrix_list.append(confusion_matrix4)

    error_rate5, confusion_matrix5 = cross_validation(dataset_random, 5)
    #print("error_rate5: ", error_rate5)
    error_rate.append(error_rate5)
    confusion_matrix_list.append(confusion_matrix5)

    error_rate6, confusion_matrix6 = cross_validation(dataset_random, 6)
    #print("error_rate6: ", error_rate6)
    error_rate.append(error_rate6)
    confusion_matrix_list.append(confusion_matrix6)

    error_rate7, confusion_matrix7 = cross_validation(dataset_random, 7)
    #print("error_rate7: ", error_rate7)
    error_rate.append(error_rate7)
    confusion_matrix_list.append(confusion_matrix7)

    error_rate8, confusion_matrix8 = cross_validation(dataset_random, 8)
    #print("error_rate8: ", error_rate8)
    error_rate.append(error_rate8)
    confusion_matrix_list.append(confusion_matrix8)

    error_rate9, confusion_matrix9 = cross_validation(dataset_random, 9)
    #print("error_rate9: ", error_rate9)
    error_rate.append(error_rate9)
    confusion_matrix_list.append(confusion_matrix9)

    error_rate10, confusion_matrix10 = cross_validation(dataset_random, 10)
    #print("error_rate10: ", error_rate10)
    error_rate.append(error_rate10)
    confusion_matrix_list.append(confusion_matrix10)

    index_min = error_rate.index(min(error_rate))
    confusion_matrix_res = confusion_matrix_list[index_min]

    sum_error = 0
    for i in range(len(error_rate)):
        sum_error = sum_error + error_rate[i]
    mean_error_rate = sum_error / len(error_rate)


    sum_error_rate_square = 0
    for i in error_rate:
        sum_error_rate_square += (i - mean_error_rate) * (i - mean_error_rate)

    cov_error_rate = sum_error_rate_square / len(error_rate)


    return mean_error_rate, cov_error_rate, min(error_rate), confusion_matrix_res

def ten_times_cross_validation(dataset):
    total_error =[]
    total_cov = []
    min_error_list =[]
    confusion_matrix_list =[]

    for i in range(10):
        print("times: ", i)
        mean_error_rate, cov_error_rate, min_error_rate_single_time, confusion_matrix_res = single_cross_validation(dataset)
        min_error_list.append(min_error_rate_single_time)
        confusion_matrix_list.append(confusion_matrix_res)
        total_error.append(mean_error_rate)
        total_cov.append(cov_error_rate)

    index_min = min_error_list.index(min(min_error_list))
    res_confusion_matrix = confusion_matrix_list[index_min]

    sum_error = 0
    for i in range(len(total_error)):
        sum_error = sum_error + total_error[i]
    mean_error_rate = sum_error / len(total_error)

    sum_error_rate_square = 0
    for i in total_cov:
        sum_error_rate_square += i

    cov_error_rate = sum_error_rate_square / len(total_cov)
    return mean_error_rate, cov_error_rate, res_confusion_matrix

def choose_best_attribute_ttt_gain_ratio(dataset, available_attribute):

    #best_information_gain = 0
    #inter = float('inf')
    decision = dataset['decision']

    gain_ratio_dic = {}
    for key in dataset:

        if (key == 'decision'):
            break

        if (key not in available_attribute):
            continue

        #attribute = dataset[key]
        gain_ratio_attribute = gain_ratio(dataset, key)
        gain_ratio_dic[key] = gain_ratio_attribute
        # best_information_gain = inter
    #print("attribute_infomation_gain_dic: ", attribute_infomation_gain_dic)
    best_attribute = max(gain_ratio_dic, key = gain_ratio_dic.get)
    #print("best_attribute: ", best_attribute)
    #print(best_attribute)
    return best_attribute

def gain_ratio(dataset, attribute_key):

    attribute = dataset[attribute_key]
    decision = dataset['decision']
    attribute_val = get_attribute_value(attribute)

    split_information = 0
    for i in attribute_val:
        if (len(dataset['decision']) == 0):
            break
        subset = get_attribute_value_subset(dataset, attribute_key, i)
        split_information += (-(len(subset['decision']) / len(dataset['decision'])) * np.log2(len(subset['decision']) / len(dataset['decision'])))
    gain = information_gain(attribute, decision)
    if (split_information == 0):
        gain_ratio = float('inf')
        return gain_ratio
    gain_ratio = gain / split_information
    return gain_ratio

def id3_ttt_gain_ratio(dataset, chosen_list):

    available_attribute =[]
    for key in dataset:
        if key not in chosen_list and key != 'decision':
            available_attribute.append(key)


    #get the key list
    total_list = []
    attribute_list = []
    feature_num = 0
    attribute_value_dic = {} #include decision
    for key in dataset:
        total_list.append(key)

        if (key != 'decision'):
            feature_num += 1
            attribute_list.append(key)
        #include decision
        attribute_value_dic[key] = get_attribute_value(dataset[key])
        """
        if(key == 'decision'):
            attribute_value_dic[key] = get_res(dataset[key])
        """
    #print("chosen list", chosen_list)
    if len(attribute_list) == len(chosen_list) or check_all_same(dataset['decision']) == True:
        count_max = float("-inf")

        max_index = 0
        for i in (attribute_value_dic['decision']):
            count = 0
            for j in (dataset['decision']):
                if j == i:
                    count += 1
            if (count > count_max):
                max_index = attribute_value_dic['decision'].index(i)
        #print(attribute_value_dic['decision'][max_index])
        if chosen_list != []:
            del chosen_list[-1]
        leaf_node = DC(atribute_num = attribute_value_dic['decision'][max_index], atribute_value=[], Children=[], LeafValue=True)
        return leaf_node

    root = DC(atribute_num = -1, atribute_value=[], Children=[], LeafValue=None)
    best_attribute = choose_best_attribute_ttt_gain_ratio(dataset, available_attribute)
    root.atribute_num = best_attribute
    root.atribute_value = attribute_value_dic[best_attribute]
    #print(root.atribute_value)
    chosen_list.append(best_attribute)

    #chosen_list_copy = deep_copy_list(chosen_list)
    #print("attribute_value_dic[attribute 4] ", attribute_value_dic['attribute_4'])
    for i in attribute_value_dic[best_attribute]:

        #dataset_copy = dataset.copy()
        best_attribute_subset = get_attribute_value_subset(dataset, best_attribute, i)
        #chosen_list = chosen_list_copy

        #print("beat_attribute_sbuset: ", best_attribute_subset)
        #print("chosen list: ",chosen_list)
        if(best_attribute_subset['decision'] == []):
            leaf_node = DC(atribute_num='positive', atribute_value=[], Children=[], LeafValue=True)
            if chosen_list != []:
                del chosen_list[-1]
            return leaf_node
        childre_node = id3_ttt_gain_ratio(best_attribute_subset, chosen_list)
        root.Children.append(childre_node)

    if chosen_list != []:
        del chosen_list[-1]
    return root

def cross_validation_ttt_gain_ratio(dataset, index_of_test_sample):


    test1_dataset, train1_dataset = split_data(dataset, index_of_test_sample)
    #print("train dataset: ", train1_dataset)
    #print("test dataset: ", test1_dataset)
    tree = id3_ttt_gain_ratio(train1_dataset,[])
    predict_res, actual_res = predict(tree, test1_dataset)

    num_wrong = 0
    for i in range(len(predict_res)):
        if(predict_res[i] != actual_res[i]):
            num_wrong += 1

    confusion_matrix = draw_confusion_matrix(predict_res, actual_res)
    error_rate = num_wrong / len(predict_res)

    return error_rate, confusion_matrix

def single_cross_validation_ttt_gain_ratio(dataset):

    dataset_random = dataset_shuffle(dataset)


    error_rate = []
    confusion_matrix_list = []

    error_rate1, confusion_matrix1 = cross_validation_ttt_gain_ratio(dataset_random, 1)
    error_rate.append(error_rate1)
    confusion_matrix_list.append(confusion_matrix1)

    #print("error_rate1: ", error_rate1)
    error_rate2, confusion_matrix2 = cross_validation_ttt_gain_ratio(dataset_random, 2)
    #print("error_rate2: ", error_rate2)
    error_rate.append(error_rate2)
    confusion_matrix_list.append(confusion_matrix2)

    error_rate3, confusion_matrix3 = cross_validation_ttt_gain_ratio(dataset_random, 3)
    #print("error_rate3: ", error_rate3)
    error_rate.append(error_rate3)
    confusion_matrix_list.append(confusion_matrix3)

    error_rate4, confusion_matrix4 = cross_validation_ttt_gain_ratio(dataset_random, 4)
    #print("error_rate4: ", error_rate4)
    error_rate.append(error_rate4)
    confusion_matrix_list.append(confusion_matrix4)

    error_rate5, confusion_matrix5 = cross_validation_ttt_gain_ratio(dataset_random, 5)
    #print("error_rate5: ", error_rate5)
    error_rate.append(error_rate5)
    confusion_matrix_list.append(confusion_matrix5)

    error_rate6, confusion_matrix6 = cross_validation_ttt_gain_ratio(dataset_random, 6)
    #print("error_rate6: ", error_rate6)
    error_rate.append(error_rate6)
    confusion_matrix_list.append(confusion_matrix6)

    error_rate7, confusion_matrix7 = cross_validation_ttt_gain_ratio(dataset_random, 7)
    #print("error_rate7: ", error_rate7)
    error_rate.append(error_rate7)
    confusion_matrix_list.append(confusion_matrix7)

    error_rate8, confusion_matrix8 = cross_validation_ttt_gain_ratio(dataset_random, 8)
    #print("error_rate8: ", error_rate8)
    error_rate.append(error_rate8)
    confusion_matrix_list.append(confusion_matrix8)

    error_rate9, confusion_matrix9 = cross_validation_ttt_gain_ratio(dataset_random, 9)
    #print("error_rate9: ", error_rate9)
    error_rate.append(error_rate9)
    confusion_matrix_list.append(confusion_matrix9)

    error_rate10, confusion_matrix10 = cross_validation_ttt_gain_ratio(dataset_random, 10)
    #print("error_rate10: ", error_rate10)
    error_rate.append(error_rate10)
    confusion_matrix_list.append(confusion_matrix10)

    index_min = error_rate.index(min(error_rate))
    confusion_matrix_res = confusion_matrix_list[index_min]

    sum_error = 0
    for i in range(len(error_rate)):
        sum_error = sum_error + error_rate[i]
    mean_error_rate = sum_error / len(error_rate)


    sum_error_rate_square = 0
    for i in error_rate:
        sum_error_rate_square += (i - mean_error_rate) * (i - mean_error_rate)

    cov_error_rate = sum_error_rate_square / len(error_rate)


    return mean_error_rate, cov_error_rate, min(error_rate), confusion_matrix_res

def ten_times_cross_validation_ttt_gain_ratio(dataset):
    total_error =[]
    total_cov = []
    min_error_list =[]
    confusion_matrix_list =[]

    for i in range(10):
        print("times: ", i)
        mean_error_rate, cov_error_rate, min_error_rate_single_time, confusion_matrix_res = single_cross_validation_ttt_gain_ratio(dataset)
        min_error_list.append(min_error_rate_single_time)
        confusion_matrix_list.append(confusion_matrix_res)
        total_error.append(mean_error_rate)
        total_cov.append(cov_error_rate)

    index_min = min_error_list.index(min(min_error_list))
    res_confusion_matrix = confusion_matrix_list[index_min]

    sum_error = 0
    for i in range(len(total_error)):
        sum_error = sum_error + total_error[i]
    mean_error_rate = sum_error / len(total_error)

    sum_error_rate_square = 0
    for i in total_cov:
        sum_error_rate_square += i

    cov_error_rate = sum_error_rate_square / len(total_cov)
    return mean_error_rate, cov_error_rate, res_confusion_matrix

def get_threshold_list(attribute):
    attribute_copy = copy.deepcopy(attribute)
    attribute_copy.sort()
    threshold_list = []
    count = 0
    while(count < len(attribute_copy ) - 1):
        threshold_ele = (attribute_copy[count] + attribute_copy[count + 1]) / 2
        threshold_list.append(threshold_ele)
        count += 2

    return threshold_list

def get_threshold(attribute, decision):
    threshold_dic = {}
    threshold_list = get_threshold_list(attribute)
    attribute_len = len(attribute)

    # find the threshold
    # information_gain_attribute = 0
    for i in range(len(threshold_list)):
        threshold = threshold_list[i]
        attribute_copy = copy.deepcopy(attribute)
        for j in range(attribute_len):

            if attribute_copy[j] <= threshold:
                attribute_copy[j] = 'low'
            else:
                attribute_copy[j] = 'high'
        information_gain_attribute = information_gain(attribute_copy, decision)
        threshold_dic[threshold_list[i]] = information_gain_attribute
    """
    for i in range(attribute_len):
        threshold = attribute[i]
        attribute_copy = copy.deepcopy(attribute)
        for j in range(attribute_len):

            if attribute_copy[j] <= threshold:
                attribute_copy[j] = 'low'
            else:
                attribute_copy[j] = 'high'
        information_gain_attribute = information_gain(attribute_copy, decision)
        threshold_dic[attribute[i]] = information_gain_attribute
    """
    # print("threshold_dic: ", threshold_dic)
    threshold = max(threshold_dic, key=threshold_dic.get)
    # print(threshold_dic)
    # print("best threshold", threshold)
    information_gain_attribute = threshold_dic[threshold]
    return threshold, information_gain_attribute


# return best_attribute
def choose_best_attribute_con(dataset, available_attribute, node):
    # best_information_gain = 0
    # inter = float('inf')
    # print("subset: ", dataset)
    # print("subset_len: ", len(dataset['decision']))
    decision = dataset['decision']

    attribute_infomation_gain_dic = {}
    dic_key_threshold = {}
    for key in dataset:

        if (key == 'decision'):
            break

        if (key not in available_attribute):
            continue

        attribute = dataset[key]
        threshold, information_gain_attribute = get_threshold(attribute, decision)
        # print("information_gain_attribute:", information_gain_attribute)
        # print("threshold", threshold)
        dic_key_threshold[key] = threshold
        # node.threshold = threshold
        attribute_infomation_gain_dic[key] = information_gain_attribute
        # best_information_gain = inter
    # print(attribute_infomation_gain_dic)
    # print("attribute_infomation_gain_dic: ", attribute_infomation_gain_dic)
    best_attribute = max(attribute_infomation_gain_dic, key=attribute_infomation_gain_dic.get)
    best_attribute_threshold = dic_key_threshold[best_attribute]
    # print("best_attribute_threshold: ", best_attribute_threshold)
    node.threshold = best_attribute_threshold
    # print(best_attribute)
    return best_attribute

# return a dic
def get_attribute_value_subset_con(dataset, attribute_key, attribute_value, node):
    # print("attribute_value: ", attribute_value)
    index_list = []
    attribute_list = dataset[attribute_key]
    attribute_list_copy = copy.deepcopy(attribute_list)
    len_attribute = len(attribute_list)

    for i in range(len_attribute):
        if attribute_list_copy[i] <= node.threshold:
            attribute_list_copy[i] = 'low'
        else:
            attribute_list_copy[i] = 'high'

    index = 0
    # get index for attribute_value

    for i in attribute_list_copy:
        if (i == attribute_value):
            index_list.append(index)
        index += 1

    sub_dataset = {}

    for key in dataset:
        sub_attribute_value = []
        for i in index_list:
            sub_attribute_value.append(dataset[key][i])

        sub_dataset[key] = sub_attribute_value

    return sub_dataset

def id3_continuous(dataset, chosen_list):
    available_attribute = []
    for key in dataset:
        if key not in chosen_list and key != 'decision':
            available_attribute.append(key)

    # get the key list
    total_list = []
    attribute_list = []
    feature_num = 0
    attribute_value_dic = {}  # include decision
    for key in dataset:
        attribute_value = []
        total_list.append(key)

        if (key != 'decision'):
            feature_num += 1
            attribute_list.append(key)
        # include decision

        for j in dataset[key]:
            attribute_value.append(j)
        attribute_value_dic[key] = attribute_value
    # print("chosen_list", chosen_list)
    if len(attribute_list) == len(chosen_list) or check_all_same(dataset['decision']) == True:
        count_max = float("-inf")

        max_index = 0
        for i in (attribute_value_dic['decision']):
            count = 0
            for j in (dataset['decision']):
                if j == i:
                    count += 1
            if (count > count_max):
                max_index = attribute_value_dic['decision'].index(i)
        # print(attribute_value_dic['decision'][max_index])
        if chosen_list != []:
            del chosen_list[-1]
        leaf_node = DC_Con(atribute_num=attribute_value_dic['decision'][max_index], Children=[[], []], LeafValue=True,
                           Threshold=-1)
        return leaf_node

    root = DC_Con(atribute_num=-1, Children=[[], []], LeafValue=None, Threshold=-1)
    best_attribute = choose_best_attribute_con(dataset, available_attribute, root)
    # print("root.threshold: ", root.threshold)
    # print(best_attribute)
    root.atribute_num = best_attribute
    # root.atribute_value = attribute_value_dic[best_attribute]
    # print(root.atribute_value)
    chosen_list.append(best_attribute)

    for i in range(len(attribute_value_dic[best_attribute])):
        # print("attribute_value_dic[best_attribute][i]: ", attribute_value_dic[best_attribute][i])
        # print("root.threshold: ", root.threshold)
        if attribute_value_dic[best_attribute][i] <= root.threshold:
            attribute_value_dic[best_attribute][i] = 'low'
        else:
            attribute_value_dic[best_attribute][i] = 'high'

    attribute_value_dic[best_attribute] = get_attribute_value(attribute_value_dic[best_attribute])

    # attribute_value_dic[best_attribute] = ['low', 'high']
    # print("attribute_value_dic: ", attribute_value_dic)
    # print("attribute_value_dic[best_attribute]: ", attribute_value_dic[best_attribute])
    for i in attribute_value_dic[best_attribute]:
        # print(i)

        if (i == 'low'):
            best_attribute_subset = get_attribute_value_subset_con(dataset, best_attribute, i, root)
            if (best_attribute_subset['decision'] == []):
                leaf_node = DC_Con(atribute_num=2, atribute_value=[], Children=[], LeafValue=True)
                if chosen_list != []:
                    del chosen_list[-1]
                return leaf_node
            # print(best_attribute_subset)
            childre_node = id3_continuous(best_attribute_subset, chosen_list)
            root.Children[0] = childre_node
        if (i == 'high'):
            best_attribute_subset = get_attribute_value_subset_con(dataset, best_attribute, i, root)
            if (best_attribute_subset['decision'] == []):
                leaf_node = DC_Con(atribute_num=2, atribute_value=[], Children=[], LeafValue=True)
                if chosen_list != []:
                    del chosen_list[-1]
                return leaf_node
            # print(best_attribute_subset)
            childre_node = id3_continuous(best_attribute_subset, chosen_list)
            root.Children[1] = childre_node
    if chosen_list != []:
            del chosen_list[-1]
    return root

# def information_gain():
def predict_con_sample(tree, sample):
    if (tree == []):
        res = 2
        return res
    if tree.Leaf_value == True:
        res = tree.atribute_num
        # print("res:", res)
        return res

    attribute = tree.atribute_num
    # print(attribute)
    branch = sample[attribute]
    # print(branch)
    # attribute_value_index = tree.atribute_value.index(branch)
    if (branch <= tree.threshold):
        sub_tree = tree.Children[0]
    else:
        sub_tree = tree.Children[1]
    # sub_tree = tree.Children[attribute_value_index]
    res = predict_con_sample(sub_tree, sample)
    return res


def predict_con(tree, test_dataset):
    test_sample = {}
    predict_res = []
    actual_res = []
    for i in range(len(test_dataset['decision'])):
        for key in test_dataset:
            if (key != 'decision'):
                test_sample[key] = test_dataset[key][i]
            else:
                actual_res.append(test_dataset[key][i])

        predict = predict_con_sample(tree, test_sample)
        predict_res.append(predict)

    return predict_res, actual_res


def cross_validation_con(dataset, index_of_test_sample):
    test1_dataset, train1_dataset = split_data(dataset, index_of_test_sample)
    tree = id3_continuous(train1_dataset, [])
    predict_res, actual_res = predict_con(tree, test1_dataset)

    num_wrong = 0
    for i in range(len(predict_res)):
        if (predict_res[i] != actual_res[i]):
            num_wrong += 1

    error_rate = num_wrong / len(predict_res)

    confusion_matrix = draw_confusion_matrix_con(predict_res, actual_res)


    return error_rate, confusion_matrix

def draw_confusion_matrix_con(predict_res, actual_res):

    confusion_matrix = np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]])
    for i in range(len(predict_res)):
        if (predict_res[i] == actual_res[i]) and (predict_res[i]  == 1):
            confusion_matrix[0][0] += 1

        if (predict_res[i] == actual_res[i]) and (predict_res[i]  == 2):
            confusion_matrix[1][1] += 1

        if (predict_res[i] == actual_res[i]) and (predict_res[i]  == 3):
            confusion_matrix[2][2] += 1

        if (predict_res[i] != actual_res[i]) and (predict_res[i]  == 1) and (actual_res[i]  == 2):
            confusion_matrix[0][1] += 1

        if (predict_res[i] != actual_res[i]) and (predict_res[i]  == 1) and (actual_res[i]  == 3):
            confusion_matrix[0][2] += 1

        if (predict_res[i] != actual_res[i]) and (predict_res[i]  == 2) and (actual_res[i]  == 1):
            confusion_matrix[0][1] += 1

        if (predict_res[i] != actual_res[i]) and (predict_res[i]  == 2) and (actual_res[i]  == 3):
            confusion_matrix[2][1] += 1

        if (predict_res[i] != actual_res[i]) and (predict_res[i]  == 3) and (actual_res[i]  == 1):
            confusion_matrix[0][2] += 1

        if (predict_res[i] != actual_res[i]) and (predict_res[i]  == 3) and (actual_res[i]  == 2):
            confusion_matrix[1][2] += 1
    confusion_matrix[0][3] = confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[0][2]
    confusion_matrix[1][3] = confusion_matrix[1][0] + confusion_matrix[1][1] + confusion_matrix[1][2]
    confusion_matrix[2][3] = confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][2]

    confusion_matrix[3][0] = confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[2][0]
    confusion_matrix[3][1] = confusion_matrix[0][1] + confusion_matrix[1][1] + confusion_matrix[2][1]
    confusion_matrix[3][2] = confusion_matrix[0][2] + confusion_matrix[1][2] + confusion_matrix[2][2]

    confusion_matrix[3][3] = confusion_matrix[0][3] + confusion_matrix[1][3] + confusion_matrix[2][3]

    return confusion_matrix

def single_cross_validation_con(dataset):
    dataset_random = dataset_shuffle(dataset)
    error_rate = []
    confusion_matrix_list = []

    error_rate1, confusion_matrix1 = cross_validation_con(dataset_random, 1)
    error_rate.append(error_rate1)
    confusion_matrix_list.append(confusion_matrix1)

    error_rate2, confusion_matrix2 = cross_validation_con(dataset_random, 2)
    error_rate.append(error_rate2)
    confusion_matrix_list.append(confusion_matrix2)

    error_rate3, confusion_matrix3 = cross_validation_con(dataset_random, 3)
    error_rate.append(error_rate3)
    confusion_matrix_list.append(confusion_matrix3)

    error_rate4, confusion_matrix4 = cross_validation_con(dataset_random, 4)
    error_rate.append(error_rate4)
    confusion_matrix_list.append(confusion_matrix4)

    error_rate5, confusion_matrix5 = cross_validation_con(dataset_random, 5)
    error_rate.append(error_rate5)
    confusion_matrix_list.append(confusion_matrix5)

    error_rate6, confusion_matrix6 = cross_validation_con(dataset_random, 6)
    error_rate.append(error_rate6)
    confusion_matrix_list.append(confusion_matrix6)

    error_rate7, confusion_matrix7 = cross_validation_con(dataset_random, 7)
    error_rate.append(error_rate7)
    confusion_matrix_list.append(confusion_matrix7)

    error_rate8, confusion_matrix8 = cross_validation_con(dataset_random, 8)
    error_rate.append(error_rate8)
    confusion_matrix_list.append(confusion_matrix8)

    error_rate9, confusion_matrix9 = cross_validation_con(dataset_random, 9)
    error_rate.append(error_rate9)
    confusion_matrix_list.append(confusion_matrix9)

    error_rate10, confusion_matrix10 = cross_validation_con(dataset_random, 10)
    error_rate.append(error_rate10)
    confusion_matrix_list.append(confusion_matrix10)

    index_min = error_rate.index(min(error_rate))
    confusion_matrix_res = confusion_matrix_list[index_min]

    sum_error = 0
    for i in range(10):
        sum_error = sum_error + error_rate[i]
    mean_error_rate = sum_error / 10

    sum_error_rate_square = 0
    for i in error_rate:
        sum_error_rate_square += (i - mean_error_rate) * (i - mean_error_rate)

    cov_error_rate = sum_error_rate_square / 10

    return mean_error_rate, cov_error_rate, min(error_rate), confusion_matrix_res

def ten_times_cross_validation_con(dataset):
    total_error =[]
    total_cov = []
    min_error_list =[]
    confusion_matrix_list =[]
    for i in range(10):
        print("times: ",i)
        mean_error_rate, cov_error_rate, min_error_rate_single_time, confusion_matrix_res = single_cross_validation_con(dataset)
        min_error_list.append(min_error_rate_single_time)
        confusion_matrix_list.append(confusion_matrix_res)
        total_error.append(mean_error_rate)
        total_cov.append(cov_error_rate)

    index_min = min_error_list.index(min(min_error_list))
    res_confusion_matrix = confusion_matrix_list[index_min]

    sum_error = 0
    for i in range(len(total_error)):
        sum_error = sum_error + total_error[i]
    mean_error_rate = sum_error / len(total_error)

    sum_error_rate_square = 0
    for i in total_cov:
        sum_error_rate_square += i

    cov_error_rate = sum_error_rate_square / len(total_cov)
    return mean_error_rate, cov_error_rate, res_confusion_matrix

def gain_ratio_con(information_gain, dataset, attribute_key,threshold):

    #attribute = dataset[attribute_key]
    #decision = dataset['decision']
    attribute_val = ['low', 'high']

    split_information = 0

    for i in attribute_val:
        if (len(dataset['decision']) == 0):
            continue
        #print(i)
        subset = get_attribute_value_subset_con_gain_ratio(dataset, attribute_key, i, threshold)
        split_information += (-(len(subset['decision']) / len(dataset['decision'])) * np.log2(len(subset['decision']) / len(dataset['decision'])))

    gain_ratio = information_gain / split_information
    return gain_ratio
# return a dic
def get_attribute_value_subset_con_gain_ratio(dataset, attribute_key, attribute_value, threshold):
    # print("attribute_value: ", attribute_value)
    index_list = []
    attribute_list = dataset[attribute_key]
    attribute_list_copy = copy.deepcopy(attribute_list)
    len_attribute = len(attribute_list)

    for i in range(len_attribute):
        if attribute_list_copy[i] <= threshold:
            attribute_list_copy[i] = 'low'
        else:
            attribute_list_copy[i] = 'high'

    index = 0
    # get index for attribute_value

    for i in attribute_list_copy:
        if (i == attribute_value):
            index_list.append(index)
        index += 1

    sub_dataset = {}

    for key in dataset:
        sub_attribute_value = []
        for i in index_list:
            sub_attribute_value.append(dataset[key][i])

        sub_dataset[key] = sub_attribute_value

    return sub_dataset

# return best_attribute
def choose_best_attribute_con_gain_ratio(dataset, available_attribute, node):
    # best_information_gain = 0
    # inter = float('inf')
    # print("subset: ", dataset)
    # print("subset_len: ", len(dataset['decision']))
    decision = dataset['decision']
    attribute_gain_ratio_dic = {}
    attribute_infomation_gain_dic = {}
    dic_key_threshold = {}

    for key in dataset:

        if (key == 'decision'):
            break

        if (key not in available_attribute):
            continue

        attribute = dataset[key]
        threshold, information_gain_attribute = get_threshold(attribute, decision)
        # print("information_gain_attribute:", information_gain_attribute)
        # print("threshold", threshold)
        dic_key_threshold[key] = threshold
        # node.threshold = threshold
        attribute_infomation_gain_dic[key] = information_gain_attribute
        # best_information_gain = inter
    # print(attribute_infomation_gain_dic)
    # print("attribute_infomation_gain_dic: ", attribute_infomation_gain_dic)
    """
    print("dic_key_threshold: ",dic_key_threshold)
    dataset_new ={}
    for key in dataset:

        if (key not in available_attribute):
            continue
        attribute_val = []

        if (key == 'decision'):
            attribute_val = dataset['decision']
            dataset_new[key] = attribute_val
            break
        for i in dataset[key]:
            if i <= dic_key_threshold[key]:
                attribute_val.append('low')
            else:
                attribute_val.append('high')
        dataset_new[key] = attribute_val
    """
    for key in attribute_infomation_gain_dic:

        information_gain = attribute_infomation_gain_dic[key]
        gain_ratio = gain_ratio_con(information_gain, dataset, key, dic_key_threshold[key])
        attribute_gain_ratio_dic[key] = gain_ratio

    best_attribute = max(attribute_gain_ratio_dic, key=attribute_gain_ratio_dic.get)
    best_attribute_threshold = dic_key_threshold[best_attribute]

    # print("best_attribute_threshold: ", best_attribute_threshold)
    node.threshold = best_attribute_threshold
    # print(best_attribute)
    return best_attribute


def id3_continuous_gain_ratio(dataset, chosen_list):
    available_attribute = []
    for key in dataset:
        if key not in chosen_list and key != 'decision':
            available_attribute.append(key)

    # get the key list
    total_list = []
    attribute_list = []
    feature_num = 0
    attribute_value_dic = {}  # include decision
    for key in dataset:
        attribute_value = []
        total_list.append(key)

        if (key != 'decision'):
            feature_num += 1
            attribute_list.append(key)
        # include decision

        for j in dataset[key]:
            attribute_value.append(j)
        attribute_value_dic[key] = attribute_value
    # print("chosen_list", chosen_list)
    if len(attribute_list) == len(chosen_list) or check_all_same(dataset['decision']) == True:
        count_max = float("-inf")

        max_index = 0
        for i in (attribute_value_dic['decision']):
            count = 0
            for j in (dataset['decision']):
                if j == i:
                    count += 1
            if (count > count_max):
                max_index = attribute_value_dic['decision'].index(i)
        # print(attribute_value_dic['decision'][max_index])
        if chosen_list != []:
            del chosen_list[-1]
        leaf_node = DC_Con(atribute_num=attribute_value_dic['decision'][max_index], Children=[[], []], LeafValue=True,
                           Threshold=-1)
        return leaf_node

    root = DC_Con(atribute_num=-1, Children=[[], []], LeafValue=None, Threshold=-1)
    best_attribute = choose_best_attribute_con_gain_ratio(dataset, available_attribute, root)
    # print("root.threshold: ", root.threshold)
    # print(best_attribute)
    root.atribute_num = best_attribute
    # root.atribute_value = attribute_value_dic[best_attribute]
    # print(root.atribute_value)
    chosen_list.append(best_attribute)

    for i in range(len(attribute_value_dic[best_attribute])):
        # print("attribute_value_dic[best_attribute][i]: ", attribute_value_dic[best_attribute][i])
        # print("root.threshold: ", root.threshold)
        if attribute_value_dic[best_attribute][i] <= root.threshold:
            attribute_value_dic[best_attribute][i] = 'low'
        else:
            attribute_value_dic[best_attribute][i] = 'high'

    attribute_value_dic[best_attribute] = get_attribute_value(attribute_value_dic[best_attribute])

    # attribute_value_dic[best_attribute] = ['low', 'high']
    # print("attribute_value_dic: ", attribute_value_dic)
    # print("attribute_value_dic[best_attribute]: ", attribute_value_dic[best_attribute])
    for i in attribute_value_dic[best_attribute]:
        # print(i)

        if (i == 'low'):
            best_attribute_subset = get_attribute_value_subset_con_gain_ratio(dataset, best_attribute, i, root.threshold)
            if (best_attribute_subset['decision'] == []):
                leaf_node = DC_Con(atribute_num=2, atribute_value=[], Children=[], LeafValue=True, Threshold=-1)
                if chosen_list != []:
                    del chosen_list[-1]
                return leaf_node
            # print(best_attribute_subset)
            childre_node = id3_continuous_gain_ratio(best_attribute_subset, chosen_list)
            root.Children[0] = childre_node
        if (i == 'high'):
            best_attribute_subset = get_attribute_value_subset_con_gain_ratio(dataset, best_attribute, i, root.threshold)
            if (best_attribute_subset['decision'] == []):
                leaf_node = DC_Con(atribute_num=2, atribute_value=[], Children=[], LeafValue=True, Threshold=-1)
                if chosen_list != []:
                    del chosen_list[-1]
                return leaf_node
            # print(best_attribute_subset)
            childre_node = id3_continuous_gain_ratio(best_attribute_subset, chosen_list)
            root.Children[1] = childre_node
    if chosen_list != []:
            del chosen_list[-1]
    return root

def cross_validation_con_gain_ratio(dataset, index_of_test_sample):
    test1_dataset, train1_dataset = split_data(dataset, index_of_test_sample)
    tree = id3_continuous_gain_ratio(train1_dataset, [])
    predict_res, actual_res = predict_con(tree, test1_dataset)

    num_wrong = 0
    for i in range(len(predict_res)):
        if (predict_res[i] != actual_res[i]):
            num_wrong += 1

    error_rate = num_wrong / len(predict_res)

    confusion_matrix = draw_confusion_matrix_con(predict_res, actual_res)


    return error_rate, confusion_matrix


def single_cross_validation_con_gain_ratio(dataset):
    dataset_random = dataset_shuffle(dataset)
    error_rate = []
    confusion_matrix_list = []

    error_rate1, confusion_matrix1 = cross_validation_con_gain_ratio(dataset_random, 1)
    error_rate.append(error_rate1)
    confusion_matrix_list.append(confusion_matrix1)

    error_rate2, confusion_matrix2 = cross_validation_con_gain_ratio(dataset_random, 2)
    error_rate.append(error_rate2)
    confusion_matrix_list.append(confusion_matrix2)

    error_rate3, confusion_matrix3 = cross_validation_con_gain_ratio(dataset_random, 3)
    error_rate.append(error_rate3)
    confusion_matrix_list.append(confusion_matrix3)

    error_rate4, confusion_matrix4 = cross_validation_con_gain_ratio(dataset_random, 4)
    error_rate.append(error_rate4)
    confusion_matrix_list.append(confusion_matrix4)

    error_rate5, confusion_matrix5 = cross_validation_con_gain_ratio(dataset_random, 5)
    error_rate.append(error_rate5)
    confusion_matrix_list.append(confusion_matrix5)

    error_rate6, confusion_matrix6 = cross_validation_con_gain_ratio(dataset_random, 6)
    error_rate.append(error_rate6)
    confusion_matrix_list.append(confusion_matrix6)

    error_rate7, confusion_matrix7 = cross_validation_con_gain_ratio(dataset_random, 7)
    error_rate.append(error_rate7)
    confusion_matrix_list.append(confusion_matrix7)

    error_rate8, confusion_matrix8 = cross_validation_con_gain_ratio(dataset_random, 8)
    error_rate.append(error_rate8)
    confusion_matrix_list.append(confusion_matrix8)

    error_rate9, confusion_matrix9 = cross_validation_con_gain_ratio(dataset_random, 9)
    error_rate.append(error_rate9)
    confusion_matrix_list.append(confusion_matrix9)

    error_rate10, confusion_matrix10 = cross_validation_con_gain_ratio(dataset_random, 10)
    error_rate.append(error_rate10)
    confusion_matrix_list.append(confusion_matrix10)

    index_min = error_rate.index(min(error_rate))
    confusion_matrix_res = confusion_matrix_list[index_min]

    sum_error = 0
    for i in range(10):
        sum_error = sum_error + error_rate[i]
    mean_error_rate = sum_error / 10

    sum_error_rate_square = 0
    for i in error_rate:
        sum_error_rate_square += (i - mean_error_rate) * (i - mean_error_rate)

    cov_error_rate = sum_error_rate_square / 10

    return mean_error_rate, cov_error_rate, min(error_rate), confusion_matrix_res



def ten_times_cross_validation_con_gain_ratio(dataset):
    total_error =[]
    total_cov = []
    min_error_list =[]
    confusion_matrix_list =[]
    for i in range(10):
        print("times: ",i)
        mean_error_rate, cov_error_rate, min_error_rate_single_time, confusion_matrix_res = single_cross_validation_con_gain_ratio(dataset)
        min_error_list.append(min_error_rate_single_time)
        confusion_matrix_list.append(confusion_matrix_res)
        total_error.append(mean_error_rate)
        total_cov.append(cov_error_rate)

    index_min = min_error_list.index(min(min_error_list))
    res_confusion_matrix = confusion_matrix_list[index_min]

    sum_error = 0
    for i in range(len(total_error)):
        sum_error = sum_error + total_error[i]
    mean_error_rate = sum_error / len(total_error)

    sum_error_rate_square = 0
    for i in total_cov:
        sum_error_rate_square += i

    cov_error_rate = sum_error_rate_square / len(total_cov)
    return mean_error_rate, cov_error_rate, res_confusion_matrix

if __name__ == "__main__":
    file_tic_tac_toe = open("/Users/mengtianao/Documents/SYDE675/a2/tic-tac-toedata.txt")

    #load the ttt to array according lines
    array_ttt = file_tic_tac_toe.readlines()
    #number of sample
    number_of_samples_ttt = len(array_ttt)

    #print(number_of_samples_ttt)

    #print(len(array_ttt[0]))
    ttt_attribute_0 = []
    ttt_attribute_1 = []
    ttt_attribute_2 = []
    ttt_attribute_3 = []
    ttt_attribute_4 = []
    ttt_attribute_5 = []
    ttt_attribute_6 = []
    ttt_attribute_7 = []
    ttt_attribute_8 = []
    ttt_decision = []
    #change each line of array_ttt to list
    for i in range(number_of_samples_ttt):
        list(array_ttt[i])

    #extract attributes to the responding attribute array
    for i in range(number_of_samples_ttt):
        count = 0
        for j in range(len(array_ttt[i])):
            if (array_ttt[i][j] != ',') and (array_ttt[i][j] != '\n'):
                if (count == 0):
                    ttt_attribute_0.append(array_ttt[i][j])
                    count += 1
                    continue

                if (count == 1):
                    ttt_attribute_1.append(array_ttt[i][j])
                    count += 1
                    continue

                if (count == 2):
                    ttt_attribute_2.append(array_ttt[i][j])
                    count += 1
                    continue

                if (count == 3):
                    ttt_attribute_3.append(array_ttt[i][j])
                    count += 1
                    continue

                if (count == 4):
                    ttt_attribute_4.append(array_ttt[i][j])
                    count += 1
                    continue

                if (count == 5):
                    ttt_attribute_5.append(array_ttt[i][j])
                    count += 1
                    continue

                if (count == 6):
                    ttt_attribute_6.append(array_ttt[i][j])
                    count += 1
                    continue

                if (count == 7):
                    ttt_attribute_7.append(array_ttt[i][j])
                    count += 1
                    continue

                if (count == 8):
                    ttt_attribute_8.append(array_ttt[i][j])
                    count += 1
                    continue
                if (count == 9):
                        ttt_decision.append(array_ttt[i][j : j+8])
                        count += 1
                        continue

    dataset_ttt= {'attribute_0': ttt_attribute_0,
                  'attribute_1': ttt_attribute_1,
                  'attribute_2': ttt_attribute_2,
                  'attribute_3': ttt_attribute_3,
                  'attribute_4': ttt_attribute_4,
                  'attribute_5': ttt_attribute_5,
                  'attribute_6': ttt_attribute_6,
                  'attribute_7': ttt_attribute_7,
                  'attribute_8': ttt_attribute_8,
                  'decision': ttt_decision,
                  }


    mean_error_rate_ttt_ig, cov_error_rate_ttt_ig, confusion_matrix_ttt_ig = ten_times_cross_validation(dataset_ttt)
    accuarcy_ttt_ig = 1 - mean_error_rate_ttt_ig
    print("accuarcy tic tac toe ig: ", accuarcy_ttt_ig)
    print("variance tic tac toe ig: ", cov_error_rate_ttt_ig)
    print(confusion_matrix_ttt_ig)



    file_wine = open("/Users/mengtianao/Documents/SYDE675/a2/winedata.txt")


    # print(dc)

    # at1_value = get_attribute_value(dataset_play_tennis['attribute_0'])
    # best_attribute = choose_best_attribute(dataset_play_tennis)
    # subset = get_attribute_value_subset(dataset_play_tennis, 'attribute_0', 'overcast')
    # print(subset)
    # print(best_attribute)
    # print(df_ttt)

    # print(values)
    # print(ttt_decision)
    # print(len(ttt_decision))

    # load the wine to array according lines
    array_wine_lines = file_wine.readlines()
    # number of sample
    # number_of_samples_wine = len(array_wine)
    wine_attribute_0 = []
    wine_attribute_1 = []
    wine_attribute_2 = []
    wine_attribute_3 = []
    wine_attribute_4 = []
    wine_attribute_5 = []
    wine_attribute_6 = []
    wine_attribute_7 = []
    wine_attribute_8 = []
    wine_attribute_9 = []
    wine_attribute_10 = []
    wine_attribute_11 = []
    wine_attribute_12 = []
    wine_decision = []

    for line in array_wine_lines:
        str_split = line.strip('\n').split(',')
        wine_attribute_0.append(str_split[1])
        wine_attribute_1.append(str_split[2])
        wine_attribute_2.append(str_split[3])
        wine_attribute_3.append(str_split[4])
        wine_attribute_4.append(str_split[5])
        wine_attribute_5.append(str_split[6])
        wine_attribute_6.append(str_split[7])
        wine_attribute_7.append(str_split[8])
        wine_attribute_8.append(str_split[9])
        wine_attribute_9.append(str_split[10])
        wine_attribute_10.append(str_split[11])
        wine_attribute_11.append(str_split[12])
        wine_attribute_12.append(str_split[13])
        wine_decision.append(str_split[0])

    wine_attribute_0 = list(map(float, wine_attribute_0))
    wine_attribute_1 = list(map(float, wine_attribute_1))
    wine_attribute_2 = list(map(float, wine_attribute_2))
    wine_attribute_3 = list(map(float, wine_attribute_3))
    wine_attribute_4 = list(map(float, wine_attribute_4))
    wine_attribute_5 = list(map(float, wine_attribute_5))
    wine_attribute_6 = list(map(float, wine_attribute_6))
    wine_attribute_7 = list(map(float, wine_attribute_7))
    wine_attribute_8 = list(map(float, wine_attribute_8))
    wine_attribute_9 = list(map(float, wine_attribute_9))
    wine_attribute_10 = list(map(float, wine_attribute_10))
    wine_attribute_11 = list(map(float, wine_attribute_11))
    wine_attribute_12 = list(map(float, wine_attribute_12))
    # print("attribute_12", wine_attribute_12)
    wine_decision = list(map(float, wine_decision))
    dataset_wine = {'attribute_0': wine_attribute_0,
                    'attribute_1': wine_attribute_1,
                    'attribute_2': wine_attribute_2,
                    'attribute_3': wine_attribute_3,
                    'attribute_4': wine_attribute_4,
                    'attribute_5': wine_attribute_5,
                    'attribute_6': wine_attribute_6,
                    'attribute_7': wine_attribute_7,
                    'attribute_8': wine_attribute_8,
                    'attribute_9': wine_attribute_9,
                    'attribute_10': wine_attribute_10,
                    'attribute_11': wine_attribute_11,
                    'attribute_12': wine_attribute_12,
                    'decision': wine_decision,
                    }

    mean_error_rate_wine_ig, cov_error_rate_wine_ig, confusion_matrix_wine_ig =  ten_times_cross_validation_con(dataset_wine)
    accuarcy_wine_ig = 1 - mean_error_rate_wine_ig
    print("accuarcy wine ig: ",accuarcy_wine_ig)
    print("variance wine ig: ", cov_error_rate_wine_ig)
    print(confusion_matrix_wine_ig)


    mean_error_rate_ttt_gain_ratio, cov_error_rate_ttt_gain_ratio, confusion_matrix_ttt_gain_ratio = ten_times_cross_validation_ttt_gain_ratio(dataset_ttt)
    accuarcy_ttt_gain_ratio = 1 - mean_error_rate_ttt_gain_ratio
    print("accuarcy tic tac toe gain ratio: ", accuarcy_ttt_gain_ratio)
    print("variance tic tac toe gain ratio: ", cov_error_rate_ttt_gain_ratio)
    print(confusion_matrix_ttt_gain_ratio)

    mean_error_rate_wine_gain_ratio, cov_error_rate_wine_gain_ratio, confusion_matrix_wine_gain_ratio =  ten_times_cross_validation_con_gain_ratio(dataset_wine)
    accuarcy_wine_gain_ratio = 1 - mean_error_rate_wine_gain_ratio
    print("accuarcy wine gain ratio: ", accuarcy_wine_gain_ratio)
    print("variance wine gain ratio: ", cov_error_rate_wine_gain_ratio)
    print(confusion_matrix_wine_gain_ratio)