# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:03:48 2020

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:47:48 2020

@author: ASUS
"""
import numpy as np
import pandas as pd
import math
import time
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, SimpleRNN, LSTM, Dropout, GRU
from numpy import nan
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, SimpleRNN, LSTM, Dropout, GRU
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
####################################################################################LOAD DATA
def load_data():
    all_data = np.load('E:/NeuNet spring99/HM/MiniPrj2\polution_dataSet.npy')
    all_data = all_data[:15000, :]
    X = np.copy(all_data[:11, :])
    for i in list(range(12,15000,12)) :
        X = np.concatenate((X, all_data[i:i+11, :].reshape(11, 8)), axis=0)
    Y = np.copy(all_data[11, :]).reshape(1,8)
    for i in list(range(23,15000,12)):
        Y = np.concatenate((Y, all_data[i, :].reshape(1,8)), axis=0)
    Y = np.copy(Y[:, 0]).reshape(1250, 1)
    X_train = np.copy(X[:11000, :])
    Y_train = np.copy(Y[:1000, :])
    X_test = np.copy(X[11000: , :])
    Y_test = np.copy(Y[1000: , :])
    return X_train, Y_train, X_test, Y_test
X_train, Y_train, X_test, Y_test = load_data()
m=np.copy( X_train[:, :]) 
###################################################################################NON 20% DATA
TO_CHANGE_NUM =2490
for i in range(0,8):
       for n in range (0,2490):
           to_change  = np.random.choice(11000, TO_CHANGE_NUM, replace=False)
           index = to_change[n]
           X_train[index][i]=nan

df=pd.DataFrame(data=X_train)  
data = df.values     

print ( ' train image dimension :' , X.ndim)
print ( ' train image  shape :' , X.shape)
print ( ' train image  type:' , X.dtype)    
for i in range(df.shape[1]):
	# count number of rows with missing values
	n_miss = df[[i]].isnull().sum()
	perc = n_miss /X_train.shape[0] * 100
	print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))
    
##############################################################################LOAD NAN 
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)    
Xtrans = imputer.transform(X_train)    
error = np.zeros((8,1))
from math import sqrt
from sklearn.metrics import mean_squared_error
n=m[:,0]
v=Xtrans[:,0]
for i in range (0,8):
    error[i,0] = sqrt(mean_squared_error(m[:,i],Xtrans[:,i]))
  ################################################################################GHESMATE 5
    
print('RMSE for column1 data is : %.3f' % error[0,0])
print('RMSE for column2 data is : %.3f' % error[1,0])

print('RMSE for column3 data is : %.3f' % error[2,0])

print('RMSE for column4 data is : %.3f' % error[3,0])

print('RMSE for column5 data is : %.3f' % error[4,0])

print('RMSE for column6 data is : %.3f' % error[5,0])

print('RMSE for column7 data is : %.3f' % error[6,0])

print('RMSE for column8 data is : %.3f' % error[7,0])
Xtrans = np.reshape(Xtrans, (int(Xtrans.shape[0]/11), 11, Xtrans.shape[1]))

################################################################################ GHESMATE6
#*****************************************************GRU NETWORK*******************
def gru(epoch): 
    X_train, Y_train, X_test, Y_test = load_data()
    X_test  = np.reshape(X_test , (int(X_test.shape[0]/11), 11, X_test.shape[1]))
    los = ['mean_squared_error', 'mean_absolute_error']
    opt = ['Adam', 'RMSprop', 'ADAgrad']
    cnt = 0
    for i in range(1):
        for j in range(1):
            cnt+=1
            use_dropout = False
            model = Sequential()
            model.add(GRU(units = 50, input_shape=(11, 8)))
            if use_dropout:
                model.add(Dropout(0.1))
            model.add((Dense(1)))
            model.compile(loss=los[i], optimizer=opt[j])
            begin = time.time()
            history = model.fit(Xtrans,Y_train,epochs = epoch
                                ,batch_size=12,verbose=2,shuffle=False,
                                validation_data=(X_test, Y_test))
            end = time.time()
            print('GRU learning time(',los[i],'and', opt[j],') is:%0.2f'%(end - begin))
            y_predicted = model.predict(X_test)
            #model.summary()
            from math import sqrt
            from sklearn.metrics import mean_squared_error
            error = sqrt(mean_squared_error(Y_test, y_predicted))
            print('RMSE for test data is : %.3f' % error)
            plt.figure('loss')
            plt.subplot(2,3,cnt)
            plt.title('loss:'+los[i]+'/optimizer:'+opt[j])
            plt.plot(history.history['loss'], label='train_loss')
            plt.plot(history.history['val_loss'], label='test_loss')
            plt.legend()
           # y_predicted = model.predict(X_test)
            plt.figure()
            plt.title('loss:'+los[i]+'/optimizer:'+opt[j])
            plt.plot(y_predicted)
            plt.plot(Y_test)
            plt.legend(['predicted', 'actual'])
            plt.figure()
            plt.plot(history.history['loss'], label='train_loss')
            plt.plot(history.history['val_loss'], label='test_loss')
            plt.title('loss:'+los[i]+'/optimizer:'+opt[j])
            plt.legend()

B=gru(100)
#lstm(2)

################################################################################ GHESMATE6
#*****************************************************lstm NETWORK*******************
def lstm(epoch): 
    X_train, Y_train, X_test, Y_test = load_data()
    #X_train = np.reshape(X_train, (int(X_train.shape[0]/11), 11, X_train.shape[1]))
    X_test  = np.reshape(X_test , (int(X_test.shape[0]/11), 11, X_test.shape[1]))
    los = ['mean_squared_error', 'mean_absolute_error']
    opt = ['adam', 'RMSprop', 'ADAgrad']
    cnt = 0
    for i in range(1):
        for j in range(1):
            cnt+=1
            model = Sequential()
            model.add(LSTM(50, input_shape=(11, 8)))
            model.add(Dense(1))
            model.compile(loss=los[i], optimizer=opt[j])
            t1 = time.time()
            history = model.fit(Xtrans, Y_train,
                                epochs=epoch, batch_size=12, verbose=2,
                                validation_data=(X_test, Y_test))
            t2 = time.time()
            print('LSTM learning time(',los[i],'and', opt[j],'):', t2-t1)
            y_predicted = model.predict(X_test)
            #model.summary()
            from math import sqrt
            from sklearn.metrics import mean_squared_error
            error = sqrt(mean_squared_error(Y_test, y_predicted))
            print('RMSE for test data is : %.3f' % error)
            plt.figure('loss')
            plt.subplot(2,3,cnt)
            plt.title('loss:'+los[i]+'/optimizer:'+opt[j])
            plt.plot(history.history['loss'], label='train_loss')
            plt.plot(history.history['val_loss'], label='test_loss')
            plt.legend()
            y_predicted = model.predict(X_test)
            plt.figure()
            plt.title('loss:'+los[i]+'/optimizer:'+opt[j])
            plt.plot(y_predicted)
            plt.plot(Y_test)
            plt.legend(['predicted', 'actual'])
            plt.figure()
            plt.plot(history.history['loss'], label='train_loss')
            plt.plot(history.history['val_loss'], label='test_loss')
            plt.title('loss:'+los[i]+'/optimizer:'+opt[j])
            plt.legend()
            
            
B=lstm(100)