import numpy as np
###########  MaxNet for using in Hamming Net   ##############
def relu(x):
    y = np.copy(x)
    a = np.where(x <= 0 )
    y[a] = 0
    return y
def w_maxnet(n, epsilon) :
    w = np.empty((n, n))
    for i in range(n):
        for j in range(n) :
            if i != j :
                w[i,j] = -epsilon
            elif i == j :
                w[i,j] = 1 
    return w
def maxnet_Q4(a, n, epsilon):   #n is dimention
    W = w_maxnet(n, epsilon)
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
    return A, maxx , q
####################################################
def w_hamming(y):
    w = y.T / 2
    return w

def Hamming_net(x, y):
    b = x.shape[1]/2
    w = w_hamming(y)
    y_in = b + np.matmul(x, w)
    n = y_in.shape[1]
    for i in range(y_in.shape[0]):
        _, maxx, q = maxnet_Q4(y_in[i, :], n, 0.15)
        print('inpute',i+1,': ', x[i, :], 'is categorized in ',y[q].reshape((6,)))
        print('there are', int(maxx), 'common elemets')





y = np.array([[1, -1, 1, -1, 1, -1],
              [-1, 1, -1, 1, -1, 1],
              [1, 1, 1, 1, 1, 1]])

x = np.array([[1,-1, 1, 1,-1,1],
              [-1, 1, 1,-1,1,-1],
              [1, 1, 1,-1,-1,-1],
              [-1,-1,-1, 1, 1,1],
              [ 1, 1, 1, 1, 1 ,1], 
              [-1,-1,1,-1,-1,-1], 
              [-1,-1,-1,1,-1,-1], 
              [1, 1,-1,-1, 1, 1], 
              [1, 1,-1, 1, 1, 1], 
              [ 1, 1, 1,-1, 1, 1]])

Hamming_net(x, y)












