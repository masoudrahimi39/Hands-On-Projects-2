# Neural Networks
# Bonus Assignment

##############################      Main      #################################

def read_train_file(file="OCR_train.txt"):
    training_data_list = []
    train_file = open(file, "r")
    for line in train_file:
        line = list(line.replace(" ", ""))
        line = [int(x) * 2 - 1 for x in line if x != "\n"]
        training_data_list.extend([line[:]])
    return training_data_list







    
###############                 Training                     ##################
    



###############          Enter your code below ...           ##################
    

def perceptron(g_train,alpha=1,theta=100) :
    w = [[0] * 63 for o in range(7)]
    net =[[0] for o in range(7)]
    summ=[[0]for o in range(7)]
    h = [0] * 7
    b = [0] * 7
    epoch = 0
    cnt = 10
    while cnt > 0  :
#    while epoch < 1000000 :
        cnt = 0
        for j in range(g_train.__len__()) :
            net =[[0] for o in range(7)]
            summ=[[0]for o in range(7)]
            target = g_train[j][-7:]
            for k in range(7):
                for i in range(63) :
                    summ[k][0] += w[k][i] * g_train[j][i]
                net[k][0] = summ[k][0] + b[k]
                if net[k][0] > theta :
                    h[k] = 1
                elif net[k][0] < -theta :
                    h[k] = -1
                else: 
                    h[k] = 0
            for k in range(7):
                if h[k] != target[k] :
                    cnt += 1
                    for i in range(63):  
                        w[k][i] = w[k][i] + alpha * g_train[j][i] * target[k]
                        b[k] = b[k] + alpha * target[k] 
        epoch += 1
    return w, b, epoch









###############          Enter your code above ...           ##################
    






###############                   Testing                    ##################



###############          Enter your code below ...           ##################
 


def test_perceptron(g_test, w, b, theta=0.02):
    net =[[0] for o in range(7)]
    summ=[[0]for o in range(7)]
    h = [0] * 7
    error = 0
    for j in range(g_test.__len__()) :
        net =[[0] for o in range(7)]
        summ=[[0]for o in range(7)]        
        target = g_test[j][-7:]
        for k in range(7):
            for i in range(63) :
                summ[k][0] += w[k][i] * g_test[j][i]
            net[k][0] = summ[k][0] + b[k]
            if net[k][0] > theta :
                h[k] = 1
            elif net[k][0] < -theta :
                h[k] = -1
            else: 
                h[k] = 0
        if h != target :
            error += 1
    return error




train_data = read_train_file(file="OCR_train.txt")
test_data = read_train_file(file="OCR_test.txt")
total = test_data.__len__()


for teta in [ 400]:
    for alfa in [.3]:
        weights, bayas, epoch = perceptron(train_data,alfa,teta)
        error = test_perceptron(test_data, weights, bayas)
        print("The Neural Network has been trained in  "+str(epoch)+" epochs & "+'\u03B1 ='+str(alfa)+' & \u03B8 ='+str(teta))
        print("Percent of Error in NN: " + str(error / total))
        print('\n')



###############          Enter your code above ...           ##################















