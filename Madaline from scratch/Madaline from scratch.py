import numpy as np
import matplotlib.pyplot as plt
######################## finding AND weights     ###############3333
def and_weights_finder(dimention, data, alpha=.1) :        # dimention = number of AND input
    b = 0.5*np.random.random_sample()                      # initial a random bayas
    weights = np.random.uniform(low=-0.25, high=0.25, size=(dimention, 1))
    cnt = 1
    epoch = 0
    while cnt > 0 :
        cnt = 0
        for q in range(data.__len__()) :
            y_in = 0
            y_in = np.dot(data[q,:-1].reshape(1, dimention), weights) + b
            if y_in >= 0 :
                y = 1
            else :
                y = -1
            if y != data[q, -1] :
                b = b + alpha * (data[q, -1] - y)
                for i in range(dimention) :
                    weights[i] = weights[i] + alpha *(data[q, -1] - y)*data[q,i]     
                cnt+=1
        epoch +=1    
    return weights,b                       

sample_4 = np.array([[-1, -1, -1, -1, -1],                    # tamam e halat haye momken baraye AND ba 4 input
                    [+1, -1, -1, -1, -1],
                    [-1, +1, -1, -1, -1],
                    [-1, -1, +1, -1, -1],
                    [-1, -1, -1, +1, -1],
                    [+1, +1, -1, -1, -1],
                    [+1, -1, +1, -1, -1],
                    [+1, -1, -1, +1, -1],
                    [-1, +1, +1, -1, -1],
                    [-1, +1, -1, +1, -1],
                    [-1, -1, +1, +1, -1],
                    [+1, +1, +1, -1, -1],
                    [+1, +1, -1, +1, -1],
                    [+1, -1, +1, +1, -1],
                    [-1, +1, +1, +1, -1],
                    [+1, +1, +1, +1, +1]])

sample_3 = np.array([[-1, -1, -1, -1],                       # tamam e halat haye momken baraye AND ba 3 input
                    [+1, -1, -1, -1],
                    [-1, +1, -1, -1],
                    [-1, -1, +1, -1],
                    [+1, +1, -1, -1],
                    [+1, -1, +1, -1],
                    [-1, +1, +1, -1],
                    [+1, +1, +1, +1]])

w_and_4, b_and_4 = and_weights_finder(4, sample_4, alpha=.01)           
w_and_3, b_and_3 = and_weights_finder(3, sample_3, alpha=.01) 
##########################      making data           ########################
## first class = the orange ones      
## second class = the green ones
blue_dot = np.array([[0,2],[1,1],[1,3.5],[2,1],[2,2],[2,3],[3,1],[3,4],[4,2],[4,4]])
class1_dim1 = np.random.normal(3, 0.2, 50)
class1_dim2 = np.random.normal(2.5, 0.3, 50)
class2_dim1 = np.random.normal(1, 0.2, 50 )
class2_dim2 = np.random.normal(2, 0.2, 50)

one = np.ones((50,))
data = np.concatenate((np.c_[class1_dim1, class1_dim2, one], np.c_[class2_dim1, class2_dim2, -one]))
 ####  third colomn of data is label.   
 ###  for orange data points, third colomn is 1 .    ### for green data points, third colomn is -1
target = np.c_[data[:, -1],-data[:, -1]]                   # one hot target
plt.title('data')
plt.plot(data[:50, 0], data[:50, 1], '*', color='orange')
plt.plot(data[50:, 0], data[50:, 1], '1', color='green')
plt.plot(blue_dot[:, 0], blue_dot[:, 1], 'o', color='blue')
plt.axis([-8, 8, -8, 8])
plt.show()
####################          normalizing data         ##############

            # 1)shiveye aval e normal kardn:
for i in range(2):
    data[:, i] = (data[:, i] - np.min(data[:, i]))/(np.max(data[:,i]) - np.min(data[:, i]))

#             2) shiveye dovom e normal kardan:
# data[:,:-1] = data[:,:-1]/np.max(data[:,:-1])
# data = np.concatenate((np.c_[class1_dim1, class1_dim2], np.c_[class2_dim1, class2_dim2]))
####################      making Madaline            ################
m1 = 4                                 ## number of neurons necessary for first class
m2 = 4                                ## number of neurons necessary for second class
m = m1 + m2                            ## number of neurons in second layer

w = np.array([[0.226, 0.03, 0.228, 0.086, -0.009, -0.07, 0.0017, 0.078],
              [0.12, -0.172, 0.01, -0.16, -0.036, -0.01, -0.05, -0.011]])
w = w.reshape((2,8))

b = np.array([[-0.86, 0.141, -0.876, 0.351, 0.109 , 0.031, 0.06, -0.1 ]])
b = b.reshape((8,1))



z_in = np.zeros((m, 1))
z = np.zeros((m, 1))
w_ = np.random.uniform(low=-0.25, high=0.25, size=(2, m))
b_ = np.random.uniform(low=-0.25, high=0.250, size=(m, 1))
y_in = np.zeros((2, 1))
y = np.zeros((2, 1))
epochs = 0 
cnt, cnt2, cnt3, cnt4 = 0,0,0,0
error = 10   # ebteda meghdar avaliye khatara qeyr e 0 dar nazar migirim ke varede halgheye while shavad
alpha = .001     
while error > 0 :                  #### while be sorat e dorost ke vaghti hich khatayi baraye hameye dade ha rokh nadahad, az hagheye an kharej mishavim. yani vaghti dar yek epoch hich khatayi nadashte bashim, az while kharej shode va learning tamam mishavad
# while epochs < 50 :                  ###### faghat baraye check kardan e masaale while ra be in shekl taghir dade am.               
    error = 0          # qabl az har epoch, error ra 0 mikonim
    res=[]            # a list for check some part of code
    correct = 0
    for q in range(data.__len__()):
        z_in = np.zeros((m, 1))
        y_in = np.zeros((2, 1))
        for i in range(m) : 
            z_in[i] = np.dot(data[q,:-1].reshape(1, 2), w_[:, i]) + b_[i]
            if z_in[i] >= 0 :
                z[i] = 1
            else :
                z[i] = -1
        y_in[0] = np.dot(z[:m1].reshape(1,m1), w_and_4) + b_and_4
        y_in[1] = np.dot(z[m1:].reshape(1,m2), w_and_4) + b_and_4
        for i in range(2):
            if y_in[i] < 0 :
                y[i] = -1
            else :
                y[i] = 1
        res.append(y)        
          ## here it checks if there is any error and calculate number of error:
        # if np.array_equal(np.transpose(y), np.matrix(target[q,:])) :  
        if np.array_equal(np.transpose(y), np.matrix(target[q,:])) :                  
            correct +=1
        else :                                         
            error +=1
        
        ### barayfor the first neuron in output layer:
        if y[0,0] != target[q, 0] :
            
            if y[0,0] == 1:                              # so target = -1. agar faghat khoroji e yeki az neuron haye miani ra be -1 tabdil konim, javab e and niz -1 mishavad
                h1 = (np.abs(z_in[:m1])).argmin()        # finding h(=index) of nearest element of z_in to zero  . ### chon baraye class e aval ast, faghat neuron haye laye miani ke be an kelas rabt daran ra baresi mikonim. be hamin dalil m1 ra dakhil kardim
                b_[h1] = b_[h1] + alpha*(-1 - z_in[h1])
                w_[0, h1] = w_[0, h1] + alpha * (-1 - z_in[h1]) *data[q,0]                    #h= closet unit of hideen layer to 0
                w_[1, h1] = w_[1, h1] + alpha * (-1 - z_in[h1]) *data[q,1] 
                cnt += 1
            if y[0,0] == -1 :                            # so target = 1. agar neuron hayi ke z_in manfi darand ra update konim, mitan omidvar bood ke khoroji hamye neuron haye laye miani marbot be on kelas niz 1 shavad va javab e and be 1 beresad.
                cnt2 +=1
                z_need_update1 = np.where(z_in[:m1]<0)[0]    ### find z_in which are negitive 
                for element in z_need_update1 :
                    w_[0, element] = w_[0, element] + alpha*(1 - z_in[element])*data[q,0]
                    w_[1, element] = w_[1, element] + alpha*(1 - z_in[element])*data[q,1]
                    b_[element] = b_[element] + alpha*(1 - z_in[element])
                    
        ### hamin revale bala baraye kelass e digar anjam midahm.::
        if y[1,0] != target[q, 1] : 
            
            if y[1,0] == 1:                    # so target = -1
                h2 = (np.abs(z_in[m1:] - 0)).argmin() + m1         #### index ra ba m1 jam mikonim ke be neuron haye marbot be kelas e mad e nazar beresim.
                b_[h2] = b_[h2] + alpha * (-1 - z_in[h2])
                w_[0, h2] = w[0, h2] + alpha * (-1 - z_in[h2]) *data[q,0]                    #h= closet unit to 0
                w_[1, h2] = w[1, h2] + alpha * (-1 - z_in[h2]) *data[q,1] 
                cnt3 +=1
            if y[1,0] == -1 :             # so target = 1
                cnt4 +=1
                z_need_update2 = np.where(z_in[m1:]<0)[0] + m1
                for element in z_need_update2 :
                    w_[0, element] = w_[0, element] + alpha*(1 - z_in[element])*data[q,0]
                    w_[1, element] = w_[1, element] + alpha*(1 - z_in[element])*data[q,1]
                    b_[element] = b_[element] + alpha*(1 - z_in[element])
    epochs +=1   
    print('error in epoch',epochs,'=',error)         
 
# print('error=', error)
# print('correct=', correct)          
########     plot boundaries     ################3
  
    



P = np.linspace(-1,4,100)
Q  = - (w[0, 0]*P + b[0])/w[1, 0]
plt.plot(P, Q, 'brown')
  
P = np.linspace(-20,20,100)
Q  = - (w[0, 1]*P + b[1])/w[1, 1]
plt.plot(P, Q, 'brown')

P = np.linspace(-20,20,100)
Q  = - (w[0, 2]*P + b[2])/w[1, 2]
plt.plot(P, Q, 'brown')
  
P = np.linspace(-20,20,100)
Q  = - (w[0, 3]*P + b[3])/w[1, 3]
plt.plot(P, Q, 'brown')
plt.show()   

################################

########################333


P = np.linspace(-20,20,100)
Q  = - (w[0, 4]*P + b[4])/w[1, 4]
plt.plot(P, Q, 'yellow')

P = np.linspace(-20,20,100)
Q  = - (w[0, 5]*P + b[5])/w[1, 5]
plt.plot(P, Q, 'yellow')

P = np.linspace(-20,20,100)
Q  = - (w[0, 6]*P + b[6])/w[1, 6]
plt.plot(P, Q, 'yellow')
  
P = np.linspace(-20,20,100)
Q  = - (w[0, 7]*P + b[7])/w[1, 7]
plt.plot(P, Q, 'yellow')
plt.show()   




######################

# P = np.linspace(-10,10,100)
# Q  =  (2.26 *P + -4.972)/0.12
# plt.plot(P, Q, 'yellow')

###########3

# P = np.linspace(-10,10,100)
# Q  =  (.787 *P + -1.02)/.11
# plt.plot(P, Q, 'b')


