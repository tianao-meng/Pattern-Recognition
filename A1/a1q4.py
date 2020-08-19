"""
q1
see in pdf file
"""
import numpy as np
import matplotlib.pyplot as plt

def hypothesis(sample , theta_cons, theta):
    
    res = (1 / (1 + (np.exp(-(theta_cons + np.dot(theta, sample))))))
    #print("res: ", res)
    return res

def cost_func(class_sample, theta_cons, theta, actual_class):
    sum = 0
    count = 0
    for sample in class_sample:
        #print("sample ", sample)
        h = hypothesis(sample, theta_cons, theta)
        #print("h ", h)
        sum += -(actual_class * np.log(h) + (1 - actual_class) * np.log(1 - h))
        #print("sum: ",sum)
        count += 1
    
    #print("sum: ",-sum/count)
    return sum / count

def gradient_descent(class_sample, theta_cons, theta,label , learning_rate):

    count = 0
    for sample in class_sample:
        actual_class = label[count]
        
        h = hypothesis(sample, theta_cons, theta)
       
        error_0 = learning_rate * (h - actual_class) * sample[0]
        #print("h ", h)
        #print("actual_class ", actual_class)
        #print("error_0 ", error_0)
        error_1 = learning_rate * (h - actual_class) * sample[1]
        #print("error_1 ", error_1)
        error_cons = learning_rate * (h - actual_class)

        theta_x = theta[0] - error_0
        theta_y = theta[1] - error_1

        theta = np.array([theta_x, theta_y])
        #print("theta", theta)
        theta_cons = theta_cons - error_cons
        cost = cost_func(class_sample, theta_cons, theta, actual_class)
        #print("cost:", cost)
        #print("cost: ", cost)
        if cost < 0.0001:
            return theta_cons, theta, cost
        
        count += 1
    return theta_cons, theta, cost

def separte(feature, label):
    data_class_0_index = []
    data_class_1_index = []
    data_class_0 = []
    data_class_1 = []
    index = 0
    for i in label:
        if i == 0:
            data_class_0_index.append(index)
            index += 1
        else :
            data_class_1_index.append(index)
            index += 1
            
    for i in data_class_0_index:
        data_class_0.append(feature[i])
    for i in data_class_1_index:
        data_class_1.append(feature[i])
    return data_class_0, data_class_1
def predict(sample, theta_cons, theta):
    

    
    h = hypothesis(sample, theta_cons, theta)
    if h > 0.5:
        return 1
    return 0
    
def normal(feature):
    sum_x = 0
    sum_y = 0
    count = 0
    sample_max = np.max(feature,axis=0)
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
if __name__ == "__main__":
    
    file = open("/Users/mengtianao/Documents/SYDE675/a1/q4/dataset3.txt")
    lines = file.readlines()
    x1 = []
    x2 = []
    label = []
    count = 0
    for line in lines:
        #print("text: ",line)
        str_split = line.strip('\n').split(',')
        #print(str_split)
        x1.append(str_split[0])
        x2.append(str_split[1])
        label.append(str_split[2])
    x1 = list(map(float, x1))
    x2 = list(map(float, x2))
    label = list(map(float, label))
    

            
    
    feature = [[]]
    x1_x2 = []
    for i in range(len(x1)):
        x1_x2 = [x1[i], x2[i]]
        if count == 0:
            feature[0] = x1_x2
            count += 1
            continue
        feature.append(x1_x2)
        count += 1
    #print("total length", len(label))
    feature = np.array(feature)
    label = np.array(label)
    
    feature_nor = normal(feature)
    data_class_0, data_class_1 = separte(feature, label)
    data_class_0 = np.array(data_class_0)
    data_class_1 = np.array(data_class_1)
    
            

   
    epoch = 5000
    cost = []
    
    np.random.seed(0)
    theta = np.random.randint(10,size = 2)
   
    #theta = np.array([0, 0])
    theta_cons = 0
    #print("label ", label[1])
    for i in range(epoch):
         #print("num ",i)
         #print("theta_cons: ",theta_cons)
        # print("theta: ",theta)
         theta_cons, theta, cost_1 = gradient_descent(feature_nor, theta_cons, theta,label , 0.001)
         cost.append(cost_1)
    epoch = np.array(range(1,epoch + 1))
    #print("cost1", cost)
    plt.title('cost-epoch') 
    plt.plot(epoch,cost)
    plt.show()
        
    """
    q3 discussion
    from this figure, we can see that at the beginning, the cost decrease a lot between 0-200, it
    get to the lowest point at around 250 epoch. after which the cost function increases, which 
    could be caused by overfitting, because of the increase of epoch.
    """
    
    predict_label_0 = []
    predict_label_1 = []
    error_num = 0
    for i in data_class_0:
        pridicted_label = predict(i, theta_cons, theta)
        if pridicted_label != 0:
            error_num += 1
            predict_label_1.append(i)
        else :
            predict_label_0.append(i)
    for i in data_class_1:
        pridicted_label = predict(i, theta_cons, theta)
        if pridicted_label != 1:
            error_num += 1
            predict_label_0.append(i)
        else :
            predict_label_1.append(i)
    
    #print("predict_label_1", predict_label_1)
    #print("predict_label_0", predict_label_0)
    
    #classify all training samples
    predict_label_0 = np.array(predict_label_0)
    predict_label_1 = np.array(predict_label_1)
    total_num = len(label)
    error_rate = error_num / total_num
    print("error rate: ", error_rate)
    #print("error_num: ", error_num)
    
    #Plot the data and show the class of the sample using different colors.
    plt.figure
    plt.title('predict result') 
    predict_class0 = plt.scatter(predict_label_0[:,0], predict_label_0[:,1],c = 'r', alpha=0.6)  
    predict_class1 = plt.scatter(predict_label_1[:,0], predict_label_1[:,1],c = 'y', alpha=0.6) 
    plt.legend((predict_class0, predict_class1), ("predict_class 0", "predict_class 1"), loc = 0)
    plt.show()
    
    plt.figure
    plt.title('actual result') 
    class0 = plt.scatter(data_class_0[:,0], data_class_0[:,1],c = 'r', alpha=0.6)  
    class1 = plt.scatter(data_class_1[:,0], data_class_1[:,1],c = 'y', alpha=0.6) 
    plt.legend((class0, class1), ("class 0", "class 1"), loc = 0)
    #Plot the Decision boundary of the classifier.
    x = feature[:,0]
    #print(theta)
    theta_x = theta[0]
    theta_y = theta[1]
    y = -theta_cons * (1 / theta_y) - theta_x * x * (1 / theta_y)
        

    plt.plot(x,y, color = 'b')
    
    plt.show()
    #print("cost ", cost)
    #print("epoch: ", epoch)
    #plt.plot(epoch, cost)
   # plt.show()
   



        
        
    
    #print ("feature: ", feature)
    #print ("x2: ", x2)
    #print ("label: ", label)
    
