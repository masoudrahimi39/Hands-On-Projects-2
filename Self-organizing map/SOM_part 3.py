from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
   
def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.copy(x_train[:1000])
    y_train = np.copy(y_train[:1000])
    x_test = np.copy(x_test[:300])
    y_test = np.copy(y_test[:300])
    x_train = x_train.reshape((1000, 28*28))
    x_test = x_test.reshape((300, 28*28))
    ###normalizing data
    x_train = x_train/np.max(x_train)
    x_test = x_test/np.max(x_test)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()
epock = 0 
itr, alpha = 100, 0.8
W = np.zeros((784, 841))
D = np.zeros(841)
while epock < itr :
    w_temp = W
    for n in range(1000):
        for j in range(841):
            D[j] = np.power(np.linalg.norm(W[:, j]-x_train[n]), 2)
        J = np.argmin(D)
        for k in range(784):
            if(J>0 and J< 840):
                
                W[k, J-1] = W[k, J-1] + alpha*(x_train[n,k] -W[k, J-1])
                W[k, J] = W[k, J] + alpha*(x_train[n,k] -W[k, J])
                W[k, J+1] = W[k, J+1] + alpha*(x_train[n,k] -W[k, J+1])
                if k!=783:
                    W[k+1, J-1] = W[k+1, J-1] + alpha*(x_train[n,k] -W[k+1, J-1])
                    W[k+1, J+1] = W[k+1, J+1] + alpha*(x_train[n,k] -W[k+1, J+1])
                    W[k+1, J] = W[k+1, J] + alpha*(x_train[n,k] -W[k+1, J])
                if k!= 0 :
                    W[k-1, J] = W[k-1, J] + alpha*(x_train[n,k] -W[k-1, J])                    
                    W[k-1, J-1] = W[k-1, J-1] + alpha*(x_train[n,k] -W[k-1, J-1])
                    W[k-1, J+1] = W[k-1, J+1] + alpha*(x_train[n,k] -W[k-1, J+1])
            elif(J==0):
                W[k, J+1] = W[k, J+1] + alpha*(x_train[n,k] -W[k, J+1])
                W[k, J] = W[k, J] + alpha*(x_train[n,k] -W[k, J])
                W[k, J-1] = W[k, J-1] + alpha*(x_train[n,k] -W[k, J-1])
                if k!=783:
                    W[k+1, J-1] = W[k+1, J-1] + alpha*(x_train[n,k] -W[k+1, J-1])
                    W[k+1, J+1] = W[k+1, J+1] + alpha*(x_train[n,k] -W[k+1, J+1])
                    W[k+1, J] = W[k+1, J] + alpha*(x_train[n,k] -W[k+1, J])
            else:
                W[k, J-1] = W[k, J-1] + alpha*(x_train[n,k] -W[k, J-1])
                W[k, J] = W[k, J] + alpha*(x_train[n,k] -W[k, J])
                if k!= 0 :
                    W[k-1, J-1] = W[k-1, J-1] + alpha*(x_train[n,k] -W[k-1, J-1])
                    W[k-1, J] = W[k-1, J] + alpha*(x_train[n,k] -W[k-1, J])


    alpha = alpha*0.5  
    epock += 1 

#w = np.copy(W)            
#winner = [[] for o in range(w.shape[1])]
#for j in range(x_test.__len__()): 
#    temp = np.zeros((1, w.shape[1]))
#    for i in range(w.shape[1]):
#        temp[0, i] = np.sum((x_test[j, :].reshape((w.shape[0], 1)) - w[:, i])**2)
#    winner[np.argmin(temp)].append(x_test[j, :])
#winners_sorted = sorted(winner, reverse = True, key=len)















