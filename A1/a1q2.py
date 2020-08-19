import numpy as np
import matplotlib.pyplot as plt
import math



x = np.linspace(0,8,300)
y = np.linspace(0,8,300)
#define function plotting decision
def plot_decision_boundary (predic_func):

    label = []
    x_min, x_max = x.min() - .5, x.max() + .5
    y_min, y_max = y.min() - .5, y.max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x_axis = xx.ravel()
    y_axis = yy.ravel()
    sample = np.array([x_axis, y_axis]).transpose()
    for i in sample:
        Z = predic_func(i, label);

    res = np.array(Z)
    res = res.reshape(xx.shape)
    plt.contourf(xx, yy, res, cmap=plt.cm.Spectral)

#class 1 prior probability, mean, covarian matrix, generate the train data
P_C1 = 0.2
mu_1 = np.array([3, 2])
Class_1_cov = np.array([[1,-1],[-1,2]])
#Class_1 = np.random.multivariate_normal(mu_1, Class_1_cov, 1000)

#class 2 prior probability  mean, covarian matrix, generate the train data
P_C2 = 0.7
mu_2 = np.array([5, 4])
Class_2_cov = np.array([[1,-1],[-1,2]])
#Class_2 = np.random.multivariate_normal(mu_2, Class_2_cov, 1000)

#class 3 prior probability  mean, covarian matrix, generate the train data
P_C3 = 0.1
mu_3 = np.array([2, 5])
Class_3_cov = np.array([[0.5,0.5],[0.5,3]])
#Class_3 = np.random.multivariate_normal(mu_3, Class_3_cov, 1000)

Class_1 = np.random.multivariate_normal(mu_1, Class_1_cov, 600)
Class_2 = np.random.multivariate_normal(mu_2, Class_2_cov, 2100)
Class_3 = np.random.multivariate_normal(mu_3, Class_3_cov, 300)

#define gaussian likelihood function

def likelihood (x, gama, mu):
    #res = (1 / (2*math.pi)**0.5) * (1 / gama ** 0.5) * np.exp((-0.5) * np.transpose((x - mu)) * np.linalg.inv(gama) * (x - mu))
    likeli = ((1 / (((2 * math.pi) **2) * (gama[0][0] * gama[1][1] - gama[0][1] * gama[1][0])) ** 0.5) * np.exp((-0.5) * np.dot(np.dot((x - mu) , np.linalg.inv(gama)) , np.transpose((x - mu)))))
    #print("likeli", likeli)
    
    #r1 = ((1 / (((2*math.pi)**0.5) * (gama[0][0]))) * np.exp((-0.5) * ((x[0] - mu[0]) ** 2) / (gama[0][0] ** 2)))
    
    #r2 = ((1 / (((2*math.pi)**0.5) * (gama[1][1]))) * np.exp((-0.5) * ((x[1] - mu[1]) ** 2) / (gama[1][1] ** 2)))
    return likeli

def posterior (prior, likelihood):
    return prior * likelihood

def plot_contour_mean(mean, covariance_mat):
    eigenvalue, eigenvector = np.linalg.eig(covariance_mat);
    eigenvalue = np.diag(eigenvalue)
    gama_inter = np.sqrt(eigenvalue)
    gama = np.linalg.inv(gama_inter)
    r = 1
    x = np.arange(-r, r+0.01, 0.1)
    y = np.sqrt(r**2 - (x)**2)
    circle = np.array([np.transpose(x),np.transpose(y)])
    std_dev_func_inter1 = np.dot(np.linalg.inv(gama),circle)
    std_dev_func = np.dot(np.linalg.inv(np.transpose(eigenvector)),std_dev_func_inter1) 
    if (mean == mu_1).all():
        plt.plot(mean[0] + std_dev_func[0],mean[1] + std_dev_func[1], color = 'w', label = 'Class 1') 
        plt.plot(mean[0] - std_dev_func[0],mean[1] - std_dev_func[1],color = 'w') 
    if (mean == mu_2).all():
        plt.plot(mean[0] + std_dev_func[0],mean[1] + std_dev_func[1], color = 'k', label = 'Class 2') 
        plt.plot(mean[0] - std_dev_func[0],mean[1] - std_dev_func[1],color = 'k') 
    if (mean == mu_3).all():
        plt.plot(mean[0] + std_dev_func[0],mean[1] + std_dev_func[1], color = 'm', label = 'Class 3') 
        plt.plot(mean[0] - std_dev_func[0],mean[1] - std_dev_func[1],color = 'm') 
    plt.scatter(mean[0], mean[1], alpha=0.6)  
    
def ML (x, label):
    likelihood_1 = likelihood(x, Class_1_cov, mu_1)
    likelihood_2 = likelihood(x, Class_2_cov, mu_2)
    likelihood_3 = likelihood(x, Class_3_cov, mu_3)
    #print(likelihood_1)
    res_ml = max(likelihood_1, likelihood_2, likelihood_3)
    if (res_ml == likelihood_1):
        label.append(0)
    if (res_ml == likelihood_2):
        label.append(1)
    if (res_ml == likelihood_3):
        label.append(2)
    return label

#define map classfier
def MAP (x, label):
    likelihood_1 = likelihood(x, Class_1_cov, mu_1)
    likelihood_2 = likelihood(x, Class_2_cov, mu_2)
    likelihood_3 = likelihood(x, Class_3_cov, mu_3)
    posterior_1 = posterior(P_C1, likelihood_1)
    posterior_2 = posterior(P_C2, likelihood_2)
    posterior_3 = posterior(P_C3, likelihood_3)
    res_map = max(posterior_1, posterior_2, posterior_3)
    if (res_map == posterior_1):
        label.append(0)
    if (res_map == posterior_2):
        label.append(1)
    if (res_map == posterior_3):
        label.append(2)
    return label
    
plot_decision_boundary(ML)
plt.title('ML method') 
plot_contour_mean(mu_1, Class_1_cov)
plot_contour_mean(mu_2, Class_2_cov)
plot_contour_mean(mu_3, Class_3_cov)
plt.legend()
plt.show()

plot_decision_boundary(MAP)
plt.title('MAP method') 
plot_contour_mean(mu_1, Class_1_cov)
plot_contour_mean(mu_2, Class_2_cov)
plot_contour_mean(mu_3, Class_3_cov)
plt.legend()
plt.show()


"""
(a) discuss
from the figure, we can get that when we use ML method, there are some difference between ml method and
map method,  because the ml method also assume the priors of these three classes are the same. From the figure
we also can get that, there are some space are classfied in class three in ml while those space is classfied in 
class 1 in map. however, the difference is subtle.
"""

#define ml classfier
def ML_q2 (x):
    likelihood_1 = likelihood(x, Class_1_cov, mu_1)
    likelihood_2 = likelihood(x, Class_2_cov, mu_2)
    likelihood_3 = likelihood(x, Class_3_cov, mu_3)
    #print(likelihood_1)
    res_ml = max(likelihood_1, likelihood_2, likelihood_3)
    if (res_ml == likelihood_1):
        return 0
    if (res_ml == likelihood_2):
        return 1
    if (res_ml == likelihood_3):
        return 2


#define map classfier
def MAP_q2 (x):
    likelihood_1 = likelihood(x, Class_1_cov, mu_1)
    likelihood_2 = likelihood(x, Class_2_cov, mu_2)
    likelihood_3 = likelihood(x, Class_3_cov, mu_3)
    posterior_1 = posterior(P_C1, likelihood_1)
    posterior_2 = posterior(P_C2, likelihood_2)
    posterior_3 = posterior(P_C3, likelihood_3)
    res_map = max(posterior_1, posterior_2, posterior_3)
    if (res_map == posterior_1):
        return 0
    if (res_map == posterior_2):
        return 1
    if (res_map == posterior_3):
        return 2

def plot_ML(Class_all, confusion_mat):
    
    count_1 = 0
    count_2 = 0
    count_3 = 0
    Class_1_ML = np.array([1,2])
    Class_2_ML = np.array([1,2])
    Class_3_ML = np.array([1,2])
    actual_class = 0
    for i in Class_all:
        for sample in i:
            predicted_class = ML_q2(sample)
            
            confusion_mat[actual_class][predicted_class]+=1

            if predicted_class == 0:
                if (count_1 == 0):
                    Class_1_ML = np.array([sample])
                    count_1 += 1
                    plt.scatter(sample[0], sample[1],color = 'b', alpha=0.6)
                    continue;
                Class_1_ML = np.append(Class_1_ML, np.array([sample]), axis = 0)
                plt.scatter(sample[0], sample[1],color = 'b',label = 'class 1', alpha=0.6)
            if predicted_class == 1:
                if (count_2 == 0):
                    Class_2_ML = np.array([sample])
                    count_2 += 1
                    plt.scatter(sample[0], sample[1],color = 'r', alpha=0.6)
                    continue;
                Class_2_ML = np.append(Class_2_ML, np.array([sample]), axis = 0)
                plt.scatter(sample[0], sample[1],color = 'r',label = 'class 2', alpha=0.6)
            if predicted_class == 2:
                if (count_3 == 0):
                    Class_3_ML = np.array([sample])
                    count_3 += 1
                    plt.scatter(sample[0], sample[1],color = 'orange', alpha=0.6) 
                    continue;
                Class_3_ML = np.append(Class_3_ML, np.array([sample]), axis = 0)
                plt.scatter(sample[0], sample[1],color = 'orange',label = 'class 3', alpha=0.6)  
        actual_class += 1            
    return Class_1_ML, Class_2_ML, Class_3_ML, count_1, count_2, count_3
def plot_MAP(Class_all, confusion_mat):
    
    count_1 = 0
    count_2 = 0
    count_3 = 0
    Class_1_MAP = np.array([1,2])
    Class_2_MAP = np.array([1,2])
    Class_3_MAP = np.array([1,2])
    actual_class = 0
    for i in Class_all:
        for sample in i:
            predicted_class = MAP_q2(sample)
            
            confusion_mat[actual_class][predicted_class]+=1
                
            if predicted_class == 0:
                if (count_1 == 0):
                    Class_1_MAP = np.array([sample])
                    count_1 += 1
                    plt.scatter(sample[0], sample[1],color = 'b',label = 'class 1', alpha=0.6)
                    continue;
                Class_1_MAP = np.append(Class_1_MAP, np.array([sample]), axis = 0)
                plt.scatter(sample[0], sample[1],color = 'b', alpha=0.6)
            if predicted_class == 1:
                if (count_2 == 0):
                    Class_2_MAP = np.array([sample])
                    count_2 += 2
                    plt.scatter(sample[0], sample[1],color = 'r',label = 'class 2', alpha=0.6)
                    continue;
                Class_2_MAP = np.append(Class_2_MAP, np.array([sample]), axis = 0)
                plt.scatter(sample[0], sample[1],color = 'r', alpha=0.6)
            if predicted_class == 2:
                if (count_3 == 0):
                    Class_3_MAP = np.array([sample])
                    count_3 += 3
                    plt.scatter(sample[0], sample[1],color = 'orange',label= 'class3', alpha=0.6)
                    continue;
                Class_3_MAP = np.append(Class_3_MAP, np.array([sample]), axis = 0)
                plt.scatter(sample[0], sample[1],color = 'orange', alpha=0.6)  
        actual_class += 1 
    return Class_1_MAP, Class_2_MAP, Class_3_MAP, count_1, count_2, count_3

Class_all = np.array([Class_1, Class_2, Class_3])
#plot the result for Ml
confusion_matrix_ml = np.array([[0,0,0],
                             [0,0,0],
                             [0,0,0]])
confusion_matrix_map = np.array([[0,0,0],
                             [0,0,0],
                             [0,0,0]])

Class_1_ML, Class_2_ML, Class_3_ML, count_1_ML, count_2_ML, count_3_ML = plot_ML(Class_all, confusion_matrix_ml)
plt.title('ML method classfy generated data') 
plt.show()
print("confusion_matrix_ml", confusion_matrix_ml)
error1_ml = confusion_matrix_ml[0][1] + confusion_matrix_ml[0][2]
error2_ml = confusion_matrix_ml[1][0] + confusion_matrix_ml[1][2]
error3_ml = confusion_matrix_ml[2][0] + confusion_matrix_ml[2][1]
p_error_ml = ((error1_ml) / 600) * P_C1 + ((error2_ml) / 2100) * P_C2 + ((error3_ml) / 300) * P_C3 
print("The probability of error ml: ", p_error_ml)
#plot the result for MAP

Class_1_MAP, Class_2_MAP, Class_3_MAP, count_1_MAP, count_2_MAP, count_3_MAP = plot_MAP(Class_all, confusion_matrix_map)
plt.title('MAP method classfy generated data') 
plt.show()
print("confusion_matrix_map", confusion_matrix_map)
error1_map = confusion_matrix_map[0][1] + confusion_matrix_map[0][2]
error2_map = confusion_matrix_map[1][0] + confusion_matrix_map[1][2]
error3_map = confusion_matrix_map[2][0] + confusion_matrix_map[2][1]
p_error_map = ((error1_map) / 600) * P_C1 + ((error2_map) / 2100) * P_C2 + ((error3_map) / 300) * P_C3 

print("The probability of error map: ", p_error_map)

        
        
    
    
    
    
    
    



    






