import numpy as np
import random

def w_modified_heb(*arg, **kwd) :    #input should be np.array
    S = list(arg)
    P = S.__len__()
    summ = np.zeros((4,4))
    for i in range(P):
        summ = summ + np.matmul(S[i],S[i].T)
    w = summ - P*np.eye(4,4)
    return w

def random_ind(list_) :  
    ind = []
    for j in range(list_.__len__()):
        
        m = random.choice(list_)
        list_.remove(m)
        ind.append(m)
    return ind

def shabake(inp, w, t) :     ## inp is a column vector
    status = 'keep on'
    rez = []
    itr = 0 
    x = np.copy(inp)
    while status == 'keep on' :
        x = np.matmul(x.T, w).T
        a = np.where(x > 0)
        x[a] = 1
        b = np.where(x < 0)
        x[b] = -1
        c = np.where(x == 0)
        x[c] = x[c]
        if np.array_equal(x, t) :
            status = 'stop'
            print('convergence')
        else :
            for element in rez :
                if np.array_equal(x, element) :
                    status = 'stop'
                    print('wrong convergence')
                else: 
                    status = 'keep on'
        rez.append(x)
        itr += 1
    return (x,'in ',itr, 'iterations')

def hopfild(x, w, t):
    y = np.copy(x)
    rez = []
    status = 'keep on'
    while status == 'keep on' :
        y_in = np.zeros((4,1))
        indeces = random_ind([0,2,3,1])
        for i in indeces :
            y_in[i] = x[i] + np.matmul(y.T, w).T[i]
            if y_in[i] > 0 :
                y[i] = 1
            elif  y_in[i] < 0 :
                y[i] = -1
            elif y_in[i] == 0 :
                y[i] = y[i]
            if np.array_equal(y, t) :
                status = 'stop'
                print('convergence')
                break
            else:
                for element in rez :
                    if np.array_equal(y, element) :
                        status = 'stop'
                        break
                        
                    else :
                        status = 'keep on'
        rez.append(y)
    return y    
##########################################################################                    
part = input('which part?(modified hebian or Hopfild)(1/2): ')
disturbed = input('Please select which kind of disturbed is there?(non/missing/mistake): ')
theta = 0
target = np.array([[1], [1], [1], [-1]])
w = w_modified_heb(target)
if disturbed == 'non' :
    S = [np.array([[1], [1], [1], [-1]])]
if disturbed == 'missing' :
    S = [np.array([[1], [0], [0], [0]]),
         np.array([[0], [1], [0], [0]]), 
         np.array([[0], [0], [1], [0]]), 
         np.array([[0], [0], [0], [-1]])]
if disturbed == 'mistake' :
    S = [np.array([[1], [-1], [-1], [1]]),
         np.array([[-1], [1], [-1], [1]]), 
         np.array([[-1], [-1], [1], [1]]), 
         np.array([[-1], [-1], [-1], [-1]])]  
    
if part == '1' :
    for element in S :
        print(shabake(element, w, target))
        
if part == '2' :
    for element in S :
        print(hopfild(element, w, target))
    








