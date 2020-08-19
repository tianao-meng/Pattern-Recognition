#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:29:56 2020

@author: mengtianao
"""

import numpy as np
import matplotlib.pyplot as plt
#construct dataset1 and dataset2 mean and covariance matrix
dataset1_mean = np.array([0,0])
dataset1_cov = np.array([[1,0],[0,1]])
dataset2_mean = np.array([0,0])
dataset2_cov = np.array([[2,-2],[-2,3]])
dataset1 = np.random.multivariate_normal(dataset1_mean, dataset1_cov, 1000)
dataset2 = np.random.multivariate_normal(dataset2_mean, dataset2_cov, 1000)

#get mean of dataset1
dt1_mean = np.mean(dataset1)
dt1_mean_column = np.mean(dataset1,0)

#get mean of dataset1
dt2_mean = np.mean(dataset2)
dt2_mean_column = np.mean(dataset2,0)

#get dataset1 covariance matrix eigenvalue and eigenvector
dt1_eigenvalue, dt1_eigenvector = np.linalg.eig(dataset1_cov);

dt1_eigenvalue = np.diag(dt1_eigenvalue)

gama1_inter1 = np.sqrt(dt1_eigenvalue)
gama1 = np.linalg.inv(gama1_inter1)

#get dataset2 covariance matrix eigenvalue and eigenvector
dt2_eigenvalue, dt2_eigenvector = np.linalg.eig(dataset2_cov);

dt2_eigenvalue = np.diag(dt2_eigenvalue)

gama2_inter1 = np.sqrt(dt2_eigenvalue)
gama2 = np.linalg.inv(gama2_inter1)

#whiten dataset1
whiten1_inter1 = np.dot(gama1,np.transpose(dt1_eigenvector))

whiten1_res = np.transpose(np.dot(whiten1_inter1,np.transpose(dataset1)))

#whiten dataset2
whiten2_inter1 = np.dot(gama2,np.transpose(dt2_eigenvector))

whiten2_res = np.transpose(np.dot(whiten2_inter1,np.transpose(dataset2)))

#cauclate dataset1 contour

r = 1
dt1_circle_center = dt1_mean_column
#print(dt1_mean_column)
#whitened dataset1 contour
dt1_x = np.arange(dt1_circle_center[0]-r, dt1_circle_center[0]+r+0.01, 0.1)
dt1_y = dt1_circle_center[1] + np.sqrt(r**2 - (dt1_x - dt1_circle_center[0])**2)
dt1_circle = np.array([np.transpose(dt1_x),np.transpose(dt1_y)])

# get dataset1 contour
dt1_first_std_dev_func_inter1 = np.dot(np.linalg.inv(gama1),dt1_circle)
dt1_first_std_dev_func = np.dot(np.linalg.inv(np.transpose(dt1_eigenvector)),dt1_first_std_dev_func_inter1)

#plot first figure
plt.figure
plt.title('First Figure') 
plt.scatter(dataset1[:,0], dataset1[:,1], alpha=0.6)  
plt.plot(dt1_first_std_dev_func[0],dt1_first_std_dev_func[1], color = 'r') 
plt.plot(dt1_first_std_dev_func[0],-dt1_first_std_dev_func[1],color = 'r') 
plt.show()

#cauclate dataset2 contour
r = 1
dt2_circle_center = dt2_mean_column
#print(dt2_mean_column)

#whitened dataset2 contour
dt2_x = np.arange(dt2_circle_center[0]-r, dt2_circle_center[0]+r+0.001, 0.1)
dt2_y = dt2_circle_center[1] + np.sqrt(r**2 - (dt2_x - dt2_circle_center[0])**2)
dt2_y_minus = - dt2_y
dt2_circle = np.array([np.transpose(dt2_x),np.transpose(dt2_y)])
dt2_circle_below = np.array([np.transpose(dt2_x),np.transpose(dt2_y_minus)])

# get dataset2 contour
dt2_first_std_dev_func_inter1 = np.dot(np.linalg.inv(gama2),dt2_circle)
dt2_first_std_dev_func = np.dot(np.linalg.inv(np.transpose(dt2_eigenvector)),dt2_first_std_dev_func_inter1)
dt2_first_std_dev_func_minus_inter1 = np.dot(np.linalg.inv(gama2),dt2_circle_below)
dt2_first_std_dev_func_minus = np.dot(np.linalg.inv(np.transpose(dt2_eigenvector)),dt2_first_std_dev_func_minus_inter1)

#plot second figure
plt.figure
plt.title('Second Figure') 
plt.scatter(dataset2[:,0], dataset2[:,1], alpha=0.6)  
plt.plot(dt2_first_std_dev_func[0],dt2_first_std_dev_func[1], color = 'r') 
plt.plot(dt2_first_std_dev_func_minus[0],dt2_first_std_dev_func_minus[1], color = 'r')
plt.show()

print(np.cov(np.transpose(dataset1)))
print('\n')
print(np.cov(np.transpose(dataset2)))

#calculate dataset1 covariance matrix
#calculate x,y variance for dataset1
dataset1_var_x = 0
count_x = 0
for dataset1_element in dataset1:
    dataset1_var_x += np.power((dataset1_element[0] - dt1_mean_column[0]),2)
    count_x += 1
dataset1_var_x = dataset1_var_x / count_x
#print(dataset1_var_x)

dataset1_var_y = 0
count_y = 0
for dataset1_element in dataset1:
    dataset1_var_y += np.power((dataset1_element[1] - dt1_mean_column[1]),2)
    count_y += 1
dataset1_var_y = dataset1_var_y / count_y
#print(dataset1_var_y)

#calculate cov(x,y) for dataset1
dataset1_cov_x_y = 0
count_x_y = 0
for dataset1_element in dataset1:
    dataset1_cov_x_y += (dataset1_element[0] - dt1_mean_column[0]) * (dataset1_element[1] - dt1_mean_column[1])
    count_x_y += 1
dataset1_cov_x_y = dataset1_cov_x_y / count_x_y
#print(dataset1_cov_x_y)

#calculate cov(y,x) for dataset2
dataset1_cov_y_x = 0
count_y_x = 0
for dataset1_element in dataset1:
    dataset1_cov_y_x += (dataset1_element[1] - dt1_mean_column[1]) * (dataset1_element[0] - dt1_mean_column[0])
    count_y_x += 1
dataset1_cov_y_x = dataset1_cov_y_x / count_y_x
#print(dataset1_cov_y_x)


dataset1_cov_res = np.array([[dataset1_var_x, dataset1_cov_x_y],
                 [dataset1_cov_y_x, dataset1_var_y]])
#print(dataset1_cov_res)


#calculate dataset2 covariance matrix
#calculate x,y variance for dataset2
dataset2_var_x = 0
count_x = 0
for dataset2_element in dataset2:
    dataset2_var_x += np.power((dataset2_element[0] - dt2_mean_column[0]),2)
    count_x += 1
dataset2_var_x = dataset2_var_x / count_x
#print(dataset2_var_x)

dataset2_var_y = 0
count_y = 0
for dataset2_element in dataset2:
    dataset2_var_y += np.power((dataset2_element[1] - dt2_mean_column[1]),2)
    count_y += 1
dataset2_var_y = dataset2_var_y / count_y
#print(dataset2_var_y)

#calculate cov(x,y) for dataset1
dataset2_cov_x_y = 0
count_x_y = 0
for dataset2_element in dataset2:
    dataset2_cov_x_y += (dataset2_element[0] - dt2_mean_column[0]) * (dataset2_element[1] - dt2_mean_column[1])
    count_x_y += 1
dataset2_cov_x_y = dataset2_cov_x_y / count_x_y
#print(dataset2_cov_x_y)

#calculate cov(y,x) for dataset2
dataset2_cov_y_x = 0
count_y_x = 0
for dataset2_element in dataset2:
    dataset2_cov_y_x += (dataset2_element[1] - dt2_mean_column[1]) * (dataset2_element[0] - dt2_mean_column[0])
    count_y_x += 1
dataset2_cov_y_x = dataset2_cov_y_x / count_y_x
#print(dataset2_cov_y_x)


dataset2_cov_res = np.array([[dataset2_var_x, dataset2_cov_x_y],
                 [dataset2_cov_y_x, dataset2_var_y]])
#print("\n")
#print(dataset2_cov_res)
    


"""
q4
we generate the dataset in Gaussian distribution, each sample in the dataset is generated randomly according to the 
covariance matrix and mean. Most sample may cause the covariance matrix in this question a.
However, there are still some outliers which lead to the covariance matrix is not the same as the one in questiona.

If we can generate enough large number of samples in our dataset, the result will be much closer to the covariance matrix
in the question a.

"""




        
    
