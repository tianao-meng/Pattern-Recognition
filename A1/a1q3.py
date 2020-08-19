import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib




def loadImageSet(filename):

    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height])

    return imgs,head


def loadLabelSet(filename):

    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)

    labelNum = head[1]
    offset = struct.calcsize('>II')

    numString = '>' + str(labelNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)

    binfile.close()
    labels = np.reshape(labels, [labelNum])

    return labels,head

#feature_num = D, sample_num = N, aim_num = d
def PCA(imgs, feature_num, sample_num, aim_num):
    #print("PCA")
    img_copy = np.copy(imgs)
    imgs_mean = np.mean(img_copy, axis=1)
    sample_cov = np.cov(img_copy)
    sample_eigenvalue, sample_eigenvector = np.linalg.eig(sample_cov)
    #print("sample_eigenvector:", sample_eigenvector)
    sorted_indices = np.argsort(sample_eigenvalue)
    W = sample_eigenvector[:,sorted_indices[:-aim_num-1:-1]]

    W = np.transpose(W)
    res = np.dot(W, img_copy)
    return res, W, imgs_mean

def detmine_dimension(imgs, feature_num, sample_num):

    POV = 0.95


    sample_cov = np.cov(imgs)
    sample_eigenvalue, sample_eigenvector = np.linalg.eig(sample_cov)
    sum_eigenvalue = np.sum(sample_eigenvalue)

    sorted_indices = np.argsort(sample_eigenvalue)
    #print(sorted_indices)
    #print("712:", sample_eigenvalue[712])
    #print("1:", sample_eigenvalue[0])

    max_eigenvalue_list = [0]
    count = 0
    for i in range(feature_num):
        total = sum(max_eigenvalue_list)
        count += 1
        if ( (total / sum_eigenvalue) > POV):
            return count
        if(i == 0):
            max_eigenvalue_list[i] = sample_eigenvalue[sorted_indices[-(i+1)]]
        max_eigenvalue_list.append(sample_eigenvalue[sorted_indices[-(i+1)]])


def reconstruction(W, res, imgs_mean):
    #print("reconstruction")
    W = np.transpose(W)
    reconstruct = np.dot(W,res)
    return reconstruct


def mean_squar_error(reconstruc_mat, sample, sample_num, feature_num):
    #print("error")
    res = 0
    reconstruc_mean = np.mean(reconstruc_mat, axis=1)
    sample_mean = np.mean(sample, axis=1)
    for i in range(feature_num):
        res += np.square(reconstruc_mean[i] - sample_mean[i])
    MSE = res / (feature_num)
    return MSE



def find_8_class(labels):
    count = 0
    for i in labels:
        if i == 8:
            return count
        count += 1


def PCA_q4(sorted_indices, aim_num):
    #print("PCA")

    #print("sample_eigenvector:", sample_eigenvector)

    eig = sorted_indices[:aim_num]
    res = np.sum(eig)
    return  res

if __name__ == "__main__":

    file1= '/Users/mengtianao/Documents/SYDE675/a1/q3/train-images.idx3-ubyte'
    file2= '/Users/mengtianao/Documents/SYDE675/a1/q3/train-labels.idx1-ubyte'

    imgs,data_head = loadImageSet(file1)
    imgs = np.transpose(imgs) # img is 784*60000
    #print(np.shape(imgs))
    sample_num = data_head[1]
    feature_num = data_head[2] * data_head[3]

    d = detmine_dimension(imgs, feature_num, sample_num)
    print("decent number: ", d)

    #q2

    dic = {}
    for i in range(20, 760, 20):
        #print("i: ", i)
        reducted_result, W, mean = PCA(imgs, feature_num, sample_num, i)
        reconstruct = reconstruction(W, reducted_result, mean)
        dic[i] = mean_squar_error(reconstruct, imgs, sample_num, feature_num)

    reducted_result, W, mean_1 = PCA(imgs, feature_num, sample_num, 1)
    reconstruct = reconstruction(W, reducted_result, mean_1)
    dic[1] = mean_squar_error(reconstruct, imgs, sample_num, feature_num)

    reducted_result, W, mean_784 = PCA(imgs, feature_num, sample_num, 784)
    reconstruct = reconstruction(W, reducted_result, mean_784)
    dic[784] = mean_squar_error(reconstruct, imgs, sample_num, feature_num)

    d = []
    MSE_arr = []
    for i in dic.keys():
        d.append(i)
        MSE_arr.append(dic[i])
    d.sort()
    MSE_arr.sort(reverse = True)
    d = np.array(d)
    MSE_arr = np.array(MSE_arr)

    dnew = np.arange(1,784,0.01)
    func = interpolate.interp1d(d,MSE_arr)
    MSE_arr_new = func(dnew)
    plt.plot(dnew, MSE_arr_new)
    plt.show()

    """
    discussion: from the figure, we can get that as the number of d increases, the MSE is decreasing.
    between 0 and 20 it decresease rapidly, after which the gap betweend different number of d is subtle.
    """

    #question 3
    labels,labels_head = loadLabelSet(file2)
    #print(labels)
    index = find_8_class(labels)
    lst = [1, 10, 50, 250, 784]
    for i in lst:
        reducted_result, W, mean_q3 = PCA(imgs, feature_num, sample_num, i)
        reconstruct = reconstruction(W, reducted_result, mean_q3)
        #print ("d: ", i)
        png_8 = reconstruct[:, index]
        #print(np.reshape(png_8,[28,28]))
        png_8 = np.reshape(png_8,[28,28])

        if (i == 1):
            matplotlib.image.imsave('1.png', png_8)
        if (i == 10):
            matplotlib.image.imsave('10.png', png_8)
        if (i == 50):
            matplotlib.image.imsave('50.png', png_8)
        if (i == 250):
            matplotlib.image.imsave('250.png', png_8)
        if (i == 784):
            matplotlib.image.imsave('784.png', png_8)


    """
    q3 discussion
    from the output of this five pictures, we can know that with the number d increases, the figure in png file is more
    accurate. and we can see that at d = 50, i think it is clear enough. thus can make sure that the 62 we get in question
    b could be a correct number of d for suitable dimension and can test the question c conclusion.
    """

    #q4
    eig = []
    sample_cov = np.cov(imgs)
    sample_eigenvalue, sample_eigenvector = np.linalg.eig(sample_cov)
    sorted_indices = np.argsort(sample_eigenvalue)
    for i in range(784):
        eigvalues = PCA_q4(sorted_indices, i)
        eig.append(eigvalues)
    y = np.array(eig)
    x = np.arange(1, 785, 1)
    plt.plot(x, y)
    plt.show()

    """
    q4 discussion
    from this figure, we can know that with the d increases, the eigenvalues increase steadily.
    from the above discussion, we can get with the sum of eigenvalues of reducted sample increase, the accuarcy
    also increase
    """
    
        
        

    

