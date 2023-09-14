from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
#def f(x):
#    if x >= 840 :
#        y = 840
#    elif x < 0:
#        
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
part = '2'
#def som(x_train, itr, R, alpha, part, m, w):
itr, R, alpha, m = 50, 0, 0.8, 841

epock = 0 
#w_intial = np.loadtxt('random_weight_Q1.csv', delimiter=',')
w = np.zeros((784, 841))
while epock < itr :
    for inputt in range(x_train.__len__()): 
        D = np.zeros((m, 1))
        for j in range(m):
            D[j] = np.dot(w[:,j]-x_train[inputt, :], w[:,j]-x_train[inputt, :])
            
        ind = np.argmin(D)
#        if part == '1' or '2' :    ### R=0 or R=1
        updating_weights = list(range(ind-R, ind+R+1))
#        elif part == '3' :
#        updating_weights = []
#        updating_weights.append([o for o in [ind-30,ind-29, ind-28, ind-1,ind,ind+1, ind+28,ind+29,ind+30]])
#        print(updating_weights)
        for element in updating_weights :
            w[:, element] = w[:, element] + alpha*(x_train[inputt, :] - w[:, element])
            c1 = alpha*(x_train[inputt, :] - w[:, element])
    alpha = alpha*0.6       ### update learining rate
    epock += 1
#return w


#part = input('please enter which part?')
#x_train, y_train, x_test, y_test = load_data()
#w_intial = np.loadtxt('random_weight_Q1.csv', delimiter=',')
############   saving random weight into CSV file   ###############
#W = np.random.rand(784, 841)     # random weights
#np.savetxt('random_weight_Q1.csv', W, delimiter=',')
#############################################################
#######################   (1)   ######################
#if part == '1' :
#    w = som(x_train, 5, 0, 0.7, part, 841, w_intial)
#
#######################################################
########################   (2)   ######################
#if part == '2' :
#    w = som(x_train, 20, 1, 0.7, part, 841, w_intial)
#
#######################################################
########################   (3)   ######################
#if part == '3' :
#    w = som(x_train, 20, 1, 0.7, part, 841, w_intial)    




#
#winner = [[] for o in range(w.shape[1])]
#for j in range(x_test.__len__()): 
#    x_ts = np.array([x_test[j, :].T,]*w.shape[1]).transpose()
#    temp = np.sum((x_ts - w)**2, axis=0)
#    winner[np.argmin(temp)].append(x_test[j, :])



#qabliii:
#winner = [[] for o in range(w.shape[1])]
#for j in range(x_test.__len__()): 
##    temp = np.zeros((1, w.shape[1]))
#    for i in range(w.shape[1]):
#        temp[0, i] = np.sum((x_test[j, :].reshape((w.shape[0], 1)) - w[:, i])**2)
#    winner[np.argmin(temp)].append(x_test[j, :])
#
#
#
#winners_sorted = sorted(winner, reverse = True, key=len)
#
#
#
#
##
#updating_weights.append(min_D_ind-30, min_D_ind-29, 
#                    min_D_ind-28, min_D_ind-1, 
#                    min_D_ind, min_D_ind+1, min_D_ind+28, min_D_ind+29,min_D_ind+30)
#
#


