import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import time

 

 ## data 2
 
#one = np.ones((100,1))
#X1 = 1 + 0.5 * (np.random.normal(0,1,100))
#X2 = 1 + 0.5 * (np.random.normal(0,1,100))
#
#Y1 = -1 + 0.5 * (np.random.normal(0,1,10))
#Y2 = -1 + 0.5 * (np.random.normal(0,1,10))

temp_data2_dimention_1 = np.c_[X1, one]
temp_data2_dimention_2 = np.c_[X2, one] 
temp_data2_dimention_1_class2 = np.c_[Y1, - np.ones((10,1))]
temp_data2_dimention_2_class2 = np.c_[Y2, - np.ones((10,1))]

data2_X1 = np.concatenate((temp_data2_dimention_1_class2, temp_data2_dimention_1), axis=0).tolist()
data2_X2 = np.concatenate((temp_data2_dimention_2_class2, temp_data2_dimention_2), axis=0).tolist()



plt.subplot(121)
plt.title('data 2')
plt.plot(X1, X2, 'bo')
plt.plot(Y1, Y2, 'ro')
plt.show()

plt.subplot(122)
plt.title('data 2')
plt.plot(X1, X2, 'bo')
plt.plot(Y1, Y2, 'ro')
plt.show()


t1_ = time.time()

             ###perceptron
net =[]
b = 0
w1 = 0
w2 = 0
theta = 0.02
alpha = 1
cnt = -1
error = 1
cnt2 = 0
while cnt !=0 :
    net.clear()
    cnt = 0
    for i in range(data2_X1.__len__()) :
        t = data2_X1[i][1]
        net.append( w1 * data2_X1[i][0] + w2 * data2_X2[i][0] + b)
        if net[i] > theta :
            h = 1
        elif net[i] < - theta :
            h = -1
        else: 
            h = 0
        error = h - t
        if error != 0 :
            w1 = w1 + alpha * data2_X1[i][0] * t
            w2 = w2 + alpha * data2_X2[i][0] * t 
            b = b + alpha * t 
            cnt += 1
    cnt2 += 1
    
    
 
p = np.linspace(-2,2,100)
q  = - (w2*p + b)/w1
plt.subplot(121)
plt.plot(p, q, 'brown', label='Perceptron & alpha=1')
plt.legend()
plt.show()
    
t2_ = time.time() - t1_
t3_ = time.time() 

              ### ADaLine


W1 = uniform(0, 0.35)
W2 = uniform(0, 0.35)            # generating small random values
bayas = uniform(0, 0.35)
ALPHA = 1                    #chosing small value
CNT = -1
error = 1
CNT2 = 0
NET = []

while CNT !=0 :
    NET.clear()
    CNT = 0
    for i in range(data2_X1.__len__()) :
        T = data2_X1[i][1]
        NET.append( W1 * data2_X1[i][0] + W2 * data2_X2[i][0] + bayas)
        if NET[i] >= 0 :
            H = 1
        else: 
            H = -1
        error = H - T
        if error != 0 :
            W1 = W1 + ALPHA * (T - NET[i]) * data2_X1[i][0] 
            W2 = W2 + ALPHA * (T - NET[i]) * data2_X2[i][0] 
            bayas = bayas + ALPHA * (T - NET[i])
            CNT += 1
    CNT2 += 1


P = np.linspace(-2,2,100)
Q  = - (W2*P + bayas)/W1
plt.subplot(122)
plt.plot(P, Q, 'brown',label='ADaLine & alpha=1')
plt.legend()
plt.show()    
    
    

t4_ = time.time() - t3_ 
    
print('perceptron time : ',t2_)
print('number of epoch for perceptron :',cnt2)
print('ADaLine time : ',t4_)
print('number of epoch for ADaLine :',CNT2)
    







