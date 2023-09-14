import numpy as np

def w_modified_heb(*arg, **kwd) :    #input should be np.array
    S = list(arg)
    P = S.__len__()
#    print(S)
    summ = np.zeros((4,4))
    for i in range(P):
        summ = summ + np.matmul(S[i],S[i].T)
    w = summ - P*np.eye(4,4)
    return w



def f(w,*arg) :         # arg is the s inputs 
    s = list(arg)
    y = []
    for i in range(len(s)) :
        y.append(np.matmul(s[i].T,w).T)
        a = np.where(y[i]>0)
        y[i][a] = 1
        b = np.where(y[i]<=0)
        y[i][b] = -1
        if np.array_equal(s[i],y[i]) :
            print('association succeeded for ', i+1,'th pattern')
            print('s: ','\n', s[i],'\n', 'y: ','\n', y[i])
        else :
            print('association failed for ', i+1, 'th pattern')
            print('s: ','\n', s[i],'\n', 'y: ','\n', y[i])
    return y        # y is a list of outputs of s

part = input('Please select which part(1,2,3,4):')

##############################################################      
############################# (1)  ###########################
if part == '1' :
    s1 = np.array([[1],[1],[1],[-1]])   
    w1 = w_modified_heb(s1)
    y1 = f(w1, s1) 

    
#############################################################
##########################    (2)   ##########################
    
if part == '2' :
    s1 = np.array([[1],[1],[1],[-1]]) 
    s2 = np.array([[1],[1],[-1],[1]])  
    w2 = w_modified_heb(s1,s2)
    y2 = f(w2, s1, s2)
    print('It can save successfully')

##########################################################
##################       (3)    ##########################
if part == '3' :
    s1 = np.array([[1],[1],[1],[-1]]) 
    s2 = np.array([[1],[1],[-1],[-1]])  
    w2 = w_modified_heb(s1, s2)
    y2 = f(w2, s1, s2)
    print('It can not save successfully due to not orthogonal input(s)')

##########################################################
##################       (4)    ##########################
if part == '4' :
    ort = input('Tell me if input vectors are mutually orthogonal or not?(y/n):')
    if ort == 'y':
        s1 = np.array([[1],[1],[1],[-1]]) 
        s2 = np.array([[1],[1],[-1],[1]])  
        s3 = np.array([[1],[-1],[1],[1]])
        w3 = w_modified_heb(s1, s2, s3)
        y3 = f(w3, s1, s2, s3)
    elif ort =='n':
        s1 = np.array([[1],[1],[1],[-1]]) 
        s2 = np.array([[1],[1],[-1],[1]])  
        s3 = np.array([[-1],[-1],[1],[1]])
        w3 = w_modified_heb(s1, s2, s3)
        y3 = f(w3, s1, s2, s3)
#    print('It can not save successfully due to not orthogonal input(s)')







