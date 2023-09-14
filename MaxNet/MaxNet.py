import numpy as np
#import matplotlib.pyplot as plt
def weight_maxnet(n, epsilon) :
    w = np.empty((n, n))
    for i in range(n):
        for j in range(n) :
            if i != j :
                w[i,j] = -epsilon
            elif i == j :
                w[i,j] = 1 
    return w
def relu(x):
    y = np.copy(x)
    a = np.where(x <= 0 )
    y[a] = 0
    return y
def maxnet(a, n, epsilon):
    W = weight_maxnet(n, epsilon)
    A = np.copy(a.T)
    a_old = np.copy(a)
    cnt = 0
    status = 'continue'
    while status == 'continue' :
        a_new = relu(np.matmul(W, a_old))
        A = np.concatenate((A, a_new.T), axis = 0)
        q = np.where(a_new > 0 )
        non_zero_nodes = a_new[q].tolist()
        if non_zero_nodes.__len__() == 1 :
            status = 'stop'
            maxx = a[q]
        else :
            a_old = np.copy(a_new)
            status = 'continue'
        cnt += 1
    return A, maxx
def maxnet_sort_maxTOmin(a, n, epsilon):
    status = 'continue'
    sort = []
    while n > 0 :
        W = weight_maxnet(n, epsilon)
        a_old = np.copy(a)
        status = 'continue'
        while status == 'continue' :
            a_new = relu(np.matmul(W, a_old))
            q = np.where(a_new > 0 )
            non_zero_nodes = a_new[q].tolist()
            if non_zero_nodes.__len__() == 1 :
                status = 'stop'
                sort.append(float(a[q]))
                a = np.delete(a,[np.argmax(a)])
                a = a.reshape(a.__len__(), 1)
            else :
                a_old = np.copy(a_new)
                status = 'continue'
        n = n-1
    return sort
def maxnet_sort_minTOmax(a, n, epsilon):
    status = 'continue'
    sort = []
    while n > 0 :
        W = weight_maxnet(n, epsilon)
        a_old = np.copy(a)
        status = 'continue'
        while status == 'continue' :
            a_new = relu(np.matmul(W, a_old))
            q = np.where(a_new > 0 )
            non_zero_nodes = a_new[q].tolist()
            if non_zero_nodes.__len__() == 1 :
                status = 'stop'
                sort.append(float(a[q]))
                a = np.delete(a,[np.argmax(a)])
                a = a.reshape(a.__len__(), 1)
            else :
                a_old = np.copy(a_new)
                status = 'continue'
        n = n-1
    sort.reverse()
    return sort
##############################################################
part = input('which part? \n1.Determining largest node\n2.Sort from max to min\n3.Sort from min to max\n(1/2/3)?: ')
#########################    (1)     #########################
if part == '1' :
    a = np.array([[1.2, 1.1, 1, 0.9, 0.95, 1.15]]).reshape(6,1)   
    A , maxx = maxnet(a, 6, 0.15)

    for j in range(A.shape[0]):
        print('a\u2081'+'('+str(j)+')='+str(format(A[j,0], '.5g'))+'\ta\u2082'+
              '('+str(j)+')='+str(format(A[j,1], '.5g'))+'\ta\u2083'+'('+str(j)+')='
              +str(format(A[j,2], '.5g'))+'\ta\u2084'+'('+str(j)+')='+str(format(A[j,3], '.5g'))
              +'\ta\u2085'+'('+str(j)+')='+str(format(A[j,4], '.5g'))+'\ta\u2086'
              +'('+str(j)+')='+str(format(A[j,5], '.5g')))
#######################################################################
############################    (2)     ############################
if part == '2' : 
    n, epsilon = 6, 0.15
    a = np.array([[1.2, 1.1, 1, 0.9, 0.95, 1.15]]).reshape(6,1)
    sort_maxTOmin = maxnet_sort_maxTOmin(a, n, epsilon)
    print('Sort from max to min:', sort_maxTOmin)
#######################################################################
############################    (3)     ############################
if part == '3' : 
    n, epsilon = 6, 0.15
    a = np.array([[1.2, 1.1, 1, 0.9, 0.95, 1.15]]).reshape(6,1)
    sort_minTOmax = maxnet_sort_minTOmax(a, n, epsilon)
    print('Sort from min to max:', sort_minTOmax)






















