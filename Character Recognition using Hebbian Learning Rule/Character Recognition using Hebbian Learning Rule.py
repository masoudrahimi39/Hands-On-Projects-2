import numpy as np
import random
import matplotlib.pyplot as plt
def hetro_ass(s,w) :      
    y = np.matmul(s.T,w).T
    a = np.where(y>0)
    b = np.where(y<=0)
    y[a] = +1
    y[b] = -1
    return y    # y is the output of the net
s_A = np.array([[-1, -1, -1, +1, -1, -1, -1],
                [-1, -1, -1, +1, -1, -1, -1],
                [-1, -1, +1, -1, +1, -1, -1],
                [-1, -1, +1, -1, +1, -1, -1],
                [-1, -1, +1, +1, +1, -1, -1],
                [-1, +1, -1, -1, -1, +1, -1],
                [-1, +1, -1, -1, -1, +1, -1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, -1, -1, -1, -1, -1, +1]])
s_A_vec = s_A.reshape((63, 1))
s_B = np.array([[+1, +1, +1, +1, +1, -1, -1],
                [+1, -1, -1, -1, -1, +1, -1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, -1, -1, -1, -1, +1, -1],
                [+1, +1, +1, +1, +1, -1, -1],
                [+1, -1, -1, -1, -1, +1, -1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, -1, -1, -1, -1, +1, -1],
                [+1, +1, +1, +1, +1, -1, -1]])
s_B_vec = s_B.reshape((63, 1))
s_C = np.array([[-1, -1, +1, +1, +1, -1, -1],
                [-1, +1, -1, -1, -1, +1, -1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, -1, -1, -1, -1, -1, -1],
                [+1, -1, -1, -1, -1, -1, -1],
                [+1, -1, -1, -1, -1, -1, -1],
                [+1, -1, -1, -1, -1, -1, +1],
                [-1, +1, -1, -1, -1, +1, -1],
                [-1, -1, +1, +1, +1, -1, -1]])
s_C_vec = s_C.reshape((63, 1))

t_A = np.array([[-1, +1, -1],
                [+1, -1, +1],
                [+1, +1, +1],
                [+1, -1, +1],
                [+1, -1, +1]])
t_A_vec = t_A.reshape((15, 1))
t_B = np.array([[+1, +1, -1],
                [+1, -1, +1],
                [+1, +1, -1],
                [+1, -1, +1],
                [+1, +1, -1]])
t_B_vec = t_B.reshape((15, 1))
t_C = np.array([[+1, +1, +1],
                [+1, -1, -1],
                [+1, -1, -1],
                [+1, -1, -1],
                [+1, +1, +1]])
t_C_vec = t_C.reshape((15, 1))

w = np.matmul(s_A_vec,t_A_vec.T) + np.matmul(s_B_vec,t_B_vec.T) + np.matmul(s_C_vec,t_C_vec.T)
indexes = list(range(0,62))
def plot_vector(a):
    if a.__len__() == 15 :
        a = a.reshape(5, 3)
        a = a * -1
    elif a.__len__() == 63 :
        a = a.reshape(9, 7)
        a = a * -1
    return plt.imshow(a, cmap='gray')
#def making_missing()
################################
disturbed_style = input('Which kind of disturbance(none/noise/missing)? (a/b/c): ') 
if disturbed_style == 'b' :
    percent = int(input('what is the percent of disturbance?(20/40):'))
elif disturbed_style == 'c' :
    percent = int(input('what is the percent of disturbance?(20/40): '))
#####################################################
#######################  (1)   ######################
if disturbed_style =='a' :     ### no disturbance
    y_A = hetro_ass(s_A_vec, w)
    y_B = hetro_ass(s_B_vec, w)
    y_C = hetro_ass(s_C_vec, w)
    cols = ['Column {}'.format(col) for col in range(1, 4)]
    plt.subplot(322)
    plt.title('y(p)\n5*3 characters\n')
    plt.axis('off')
    plot_vector(y_A)
    plt.subplot(321)
    plt.title('s(p)\n9*7 characters\n')
    plt.axis('off')
    plot_vector(s_A_vec)
    plt.subplot(324)
    plt.axis('off')
    plot_vector(y_B)
    plt.subplot(323)
    plt.axis('off')
    plot_vector(s_B_vec)
    plt.subplot(326)
    plt.axis('off')
    plot_vector(y_C)
    plt.subplot(325)
    plt.axis('off')
    plot_vector(s_C_vec)
####################################################
######################   (2)   #####################
k = 10000
if disturbed_style == 'b' :    ### there is noise
    if percent == 20 :
        crt = 0
        for j in range(k) :    # k : number of itration
            s_A_vec_n = np.copy(s_A_vec)
            s_B_vec_n = np.copy(s_B_vec)
            s_C_vec_n = np.copy(s_C_vec)
            rand_ind = random.sample(indexes,int(63*0.2))
            for i in rand_ind :
                s_A_vec_n[i,0] = s_A_vec_n[i,0] * (-1)
                s_B_vec_n[i,0] = s_B_vec_n[i,0] * (-1)
                s_C_vec_n[i,0] = s_C_vec_n[i,0] * (-1)
            y_A = hetro_ass(s_A_vec_n, w)
            y_B = hetro_ass(s_B_vec_n, w)
            y_C = hetro_ass(s_C_vec_n, w)
            if np.array_equal(y_A, t_A_vec) :
                if np.array_equal(y_B, t_B_vec) :
                    if np.array_equal(y_C, t_C_vec) :
                        crt+=1
        print('percent of correct assosiation in 20% mistake : ',100*crt/k )
    elif percent == 40 :
        crt = 0
        for j in range(k) :    # k : number of itration
            s_A_vec_n = np.copy(s_A_vec)
            s_B_vec_n = np.copy(s_B_vec)
            s_C_vec_n = np.copy(s_C_vec)
            rand_ind = random.sample(indexes,int(63*0.4))
            for i in rand_ind :
                s_A_vec_n[i,0] = s_A_vec_n[i,0] * (-1)
                s_B_vec_n[i,0] = s_B_vec_n[i,0] * (-1)
                s_C_vec_n[i,0] = s_C_vec_n[i,0] * (-1)
            y_A = hetro_ass(s_A_vec_n, w)
            y_B = hetro_ass(s_B_vec_n, w)
            y_C = hetro_ass(s_C_vec_n, w)
            if np.array_equal(y_A, t_A_vec) :
                if np.array_equal(y_B, t_B_vec) :
                    if np.array_equal(y_C, t_C_vec) :
                        crt+=1
#            else 
        print('percent of correct assosiation in 40% mistake: ',crt/k )
################################################################
#############################    (3)     ######################
if disturbed_style == 'c' :    ### there's missing data
    if percent == 20 :
        crt = 0
        for j in range(k) :    # k : number of itration
            s_A_vec_n = np.copy(s_A_vec)
            s_B_vec_n = np.copy(s_B_vec)
            s_C_vec_n = np.copy(s_C_vec)
            rand_ind = random.sample(indexes,int(63*0.2))
            for i in rand_ind :
                s_A_vec_n[i] = s_A_vec_n[i] * (0)
                s_B_vec_n[i] = s_B_vec_n[i] * (0)
                s_C_vec_n[i] = s_C_vec_n[i] * (0)
            y_A = hetro_ass(s_A_vec_n,w)
            y_B = hetro_ass(s_B_vec_n,w)
            y_C = hetro_ass(s_C_vec_n,w)
            if np.array_equal(y_A, t_A_vec) :
                if np.array_equal(y_B, t_B_vec) :
                    if np.array_equal(y_C, t_C_vec) :
                        crt+=1
        print('percent of correct assosiation in 20% missing: ',crt/k )

    elif percent == 40 :
        crt = 0
        for j in range(k) :    # k : number of itration
            s_A_vec_n = np.copy(s_A_vec)
            s_B_vec_n = np.copy(s_B_vec)
            s_C_vec_n = np.copy(s_C_vec)
            rand_ind = random.sample(indexes,int(63*0.4))
            for i in rand_ind :
                s_A_vec_n[i] = s_A_vec_n[i] * (0)
                s_B_vec_n[i] = s_B_vec_n[i] * (0)
                s_C_vec_n[i] = s_C_vec_n[i] * (0)
            y_A = hetro_ass(s_A_vec_n,w)
            y_B = hetro_ass(s_B_vec_n,w)
            y_C = hetro_ass(s_C_vec_n,w)
            if np.array_equal(y_A, t_A_vec) :
                if np.array_equal(y_B, t_B_vec) :
                    if np.array_equal(y_C, t_C_vec) :
                        crt+=1
        print('percent of correct assosiation in 40% missing: ',crt/k )
    



