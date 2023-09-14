import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
def w_mod_heb(s1, t1, s2, t2) :    #input should be s1, t1, s2, t2 in np.array
    w1 = np.matmul(s1, t1.T)         # s & t are column vectors 
    w2 = np.matmul(s2, t2.T)
    w = w1 + w2
    return w

def plot_vector(a):
    if a.__len__() == 280 :
        a = a.reshape(8, 35)
        a = a * -1
    elif a.__len__() == 288 :
        a = a.reshape(18, 16)
        a = a * -1
    return plt.imshow(a, cmap='gray')

def making_y(x, w) :        
    temp = np.matmul(x.T, w).T
    y = np.zeros(np.shape(temp))
    y_in = np.matmul(x.T, w).T
    a = np.where(y_in > 0)
    y[a] = 1
    b = np.where(y_in < 0)
    y[b] = -1 
    c = np.where(y_in == 0)
    y[c] = y[c]
    y = y.reshape(8,35)
    y = y * -1
    return plt.imshow(y, cmap='gray')
    
def making_x(y, w) :
    temp = np.matmul(y.T, w.T).T
    x = np.zeros(np.shape(temp))
    x_in = np.matmul(y.T, w.T).T
    a = np.where(x_in > 0)
    x[a] = 1
    b = np.where(x_in < 0)
    x[b] = -1 
    c = np.where(x_in == 0)
    x[c] = y[c]
    x = x.reshape(18,16)
    x = x * -1
    return plt.imshow(x, cmap='gray')

def x_to_y(x, y, w, s, t) :    # x, y, s, t are column vectors
    plt.subplot(221)
    plt.title('x : noisy of s(p) \n')
    plt.axis('off')
    plot_vector(x)
    plt.subplot(222)
    plt.title('y : noisy of t(p) \n')
    plt.axis('off')
    plot_vector(y)
    y_in = np.zeros(np.shape(y))
    x_in = np.zeros(np.shape(x))
    status = 'keep on'
    itr = 0
    while status == 'keep on' :
        y_in = np.matmul(x.T, w).T
        a = np.where(y_in > 0)
        y[a] = 1
        b = np.where(y_in < 0)
        y[b] = -1 
        c = np.where(y_in == 0)
        y[c] = y[c]
        if np.array_equal(y, t) and np.array_equal(x, s):
            status = 'convergence'
        else :
            x_in = np.matmul(y.T, w.T).T
            d = np.where(x_in > 0)
            x[d] = 1
            e = np.where(x_in < 0)
            x[e] = -1 
            f = np.where(x_in == 0)
            x[f] = x[f]
            if np.array_equal(x, s) and np.array_equal(x, s) :
                status = 'convergence'
        itr += 1
    plt.subplot(223)
    plt.title('converged x \n')
    plt.axis('off')
    plot_vector(x)
    plt.subplot(224)
    plt.title('converged y \n')
    plt.axis('off')
    plot_vector(y)
    return itr
    
def noisy(a) :    ## 30% of 'a' will be changed     ## 'a' is a column vector
    indexes = random.sample(list(range(a.__len__())), int(a.__len__()*0.3))
    np.put(a, indexes , [-a[o] for o in indexes] )
    return a
    
part = input('Which part?(1/2) :')    
########################################################################  
#####################    Reading input and putput vectors    ###############
Splane = cv2.imread('plane.png',0)
Splane = (2*np.sign(-1*Splane)+np.ones([18,16])).astype(int)
Tplane = cv2.imread('plane2.png',0)
Tplane = (2*np.sign(-1*Tplane)+np.ones([8,35])).astype(int)
Stank = cv2.imread('tank.png',0)
Stank = (2*np.sign(-1*Stank)+np.ones([18,16])).astype(int)
Ttank = cv2.imread('tank2.png',0)
Ttank = (2*np.sign(-1*Ttank)+np.ones([8,35])).astype(int)
     #### converting plane & tank inputs & outputs to vector
     ### Sp:s_plane   /   Tp: t_plane   /  St: s_tank   /   Tt: t_tank
Sp = Splane.reshape(np.size(Splane),1)
Tp = Tplane.reshape(np.size(Tplane),1)
St = Stank.reshape(np.size(Stank),1)
Tt = Ttank.reshape(np.size(Ttank),1)
##########################################################################
###############################    (1)    ################################
if part == '1' :
    w = w_mod_heb(Sp, Tp, St, Tt) 
    plt.subplot(221)
    plt.title('making t(plane) form s(plane) \n')
    making_y(Sp, w)
    plt.axis('off')
    plt.subplot(223)
    plt.title('making s(plane) form t(plane)')
    making_x(Tp, w)
    plt.axis('off')
    plt.subplot(222)
    plt.title('making t(tank) form s(tank) \n')
    making_y(St, w)
    plt.axis('off')
    plt.subplot(224)
    plt.title('making s(tank) form t(tank)')
    making_x(Tt, w)
    plt.axis('off')
    plt.show()
##########################################################################
#############################     (2)      ###############################
if part == '2' :
        ### noisy data 30% mistake
    Splane_n = cv2.imread('plane.png',0)
    Splane_n = (2*np.sign(-1*Splane_n)+np.ones([18,16])).astype(int)
    Tplane_n = cv2.imread('plane2.png',0)
    Tplane_n = (2*np.sign(-1*Tplane_n)+np.ones([8,35])).astype(int)
    Stank_n = cv2.imread('tank.png',0)
    Stank_n = (2*np.sign(-1*Stank_n)+np.ones([18,16])).astype(int)
    Ttank_n = cv2.imread('tank2.png',0)
    Ttank_n = (2*np.sign(-1*Ttank_n)+np.ones([8,35])).astype(int)
    Sp_n = Splane_n.reshape(np.size(Splane_n), 1)
    Tp_n = Tplane_n.reshape(np.size(Tplane_n), 1)
    St_n = Stank_n.reshape(np.size(Stank_n), 1)
    Tt_n = Ttank_n.reshape(np.size(Ttank_n), 1)
    Sp_n = noisy(Sp_n)
    Tp_n = noisy(Tp_n)
    St_n = noisy(St_n)
    Tt_n = noisy(Tt_n)

    w = w_mod_heb(Sp, Tp, St, Tt) 
    plt.figure('Plane')
    x_to_y(Sp_n, Tp_n, w, Sp, Tp)
    plt.figure('Tank')
    x_to_y(St_n, Tt_n, w, St, Tt)
