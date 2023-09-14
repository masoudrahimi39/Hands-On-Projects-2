import numpy as np
import matplotlib.pyplot as plt
def saturated_Relu(x, x_max=2):
    y = np.copy(x)
    a = np.where(x <= 0)
    y[a] = 0
    b = np.where(x > x_max)
    y[b] = x_max
    return y
def f(x, n):       
    if x<0:
        y = 0
    elif x>=n :
        y = n
    else:
        y = x
    return y
def mexican_hat(s, R1, R2, C1, C2,  n, t_max) :
    t = 0
    x = np.copy(s)
    A = np.copy(x.T)
    while t < t_max :
        x_old = np.copy(x)
        positive = []
        temp=[] 
        negative= []
        for i in range(x_old.__len__()):
            summ = 0
            positive = list(range(f(i-R1, n),f(i+R1+1, n)))
            temp = list(range(f(i-R2, n),f(i+R2+1, n)))
            negative = [o for o in temp if o not in positive]
            for pos in positive :
                summ = summ + C1*x_old[pos]
#                print(i, x_old[pos])
            for neg in negative :
#                print(i, x_old[neg])
                summ = summ + C2* x_old[neg]
#            print(summ)
            x[i] = summ
        x = saturated_Relu(x) 
        A = np.concatenate((A, x.T), axis=0)
        t += 1
    return A 
def determining_C_part2(n, t_max):
    R1, R2 = 1, 3
    s_det = np.array([ 0.27, 0.35, 0.44, 0.58, 0.66, 0.77,
                  0.4, 0.32, 0.20, 0.15, 0.08 ]).reshape((11,1))
    cnt = 1
    for o in [[0.6, -0.2], [0.6, -0.3], [0.7, -0.2], [0.7, -0.3], [0.7, -0.4], [0.8, -0.4],[0.8, -0.5]]:
        C1 = o[0]
        C2 = o[1]
        bb = mexican_hat(s_det,R1, R2, C1, C2, n, t_max)
        plt.subplot(3, 3, cnt)
        for i in range(bb.shape[0]):
            plt.plot(bb[i, :])
        plt.title('C1: '+str(C1)+' C2: '+str(C2))
        plt.legend(['t=0', 't=1', 't=2', 't=3', 't=4'])
        cnt+=1
        plt.show()

part = input('which part?\n1. R\u2081=0, R\u2082=∞\n2. R\u2081=1, R\u2082=3:\n3. determining appropriate C\u2081 and C\u2082\n (1/2/3)?:' )
############ emtahanii ba example e jozve #########################
#R1, R2, C1, C2, t_max, n =1, 2, 0.6, -0.4, 2, 7
#s1 = np.array([0, 0.5, 0.8, 1, 0.8, 0.5, 0]).reshape((7,1))
#emt = mexican_hat(s1,R1, R2, C1, C2, n, t_max)
#plt.figure()
#for i in range(emt.shape[0]):
#    plt.plot(emt[i, :])
#plt.legend(['t=0', 't=1', 't=2', 't=3'])
#plt.show()
###################  alef ######################

if part == '1' :
    n = 11
    t_max = 3
    R1, R2, C1, C2 = 0, 100, 0.9, -0.2
    s = np.array([ 0.27, 0.35, 0.44, 0.58, 0.66, 0.77,
                  0.4, 0.32, 0.20, 0.15, 0.08 ]).reshape((11,1))
    
    bb = mexican_hat(s,R1, R2, C1, C2, n, t_max)
    plt.figure()
    for i in range(bb.shape[0]):
        plt.plot(bb[i, :])
    plt.title('C\u2081= '+str(C1)+'   C\u2082= '+str(C2))
    plt.legend(['t=0', 't=1', 't=2', 't=3', 't=4'])
    plt.show()

##############################################################
####################### part 2   #############################
if part == '2' :
    n = 11
    t_max = 3
    R1, R2, C1, C2 = 1, 3, 0.7, -0.4
    s = np.array([ 0.27, 0.35, 0.44, 0.58, 0.66, 0.77,
                  0.4, 0.32, 0.20, 0.15, 0.08 ]).reshape((11,1))
    
    bb = mexican_hat(s, R1, R2, C1, C2, n, t_max)
    plt.figure()
    for i in range(bb.shape[0]):
        plt.plot(bb[i, :])
    plt.title('C\u2081= '+str(C1)+'   C\u2082= '+str(C2))
    plt.legend(['t=0', 't=1', 't=2', 't=3', 't=4'])
    plt.show()

########################################################################
######################    part3 : determining C1 , C2 if R1=1, R2=3 ###############
if part == '3' : 
    n , t_max= 11, 3
    determining_C_part2(n, t_max)



