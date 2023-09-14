#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adagrad
from keras.losses import mean_squared_error
# from sklearn.preprocessing import StandardScaler


def plot_(network_history) :
    history = network_history.history
    losses = history['loss']
    losses_val = history['val_loss']
    
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.plot(losses)
    plt.plot(losses_val)
    plt.legend(['loss', 'val_loss']) 
    
    
    
    
########## import data   ##########

#_______________________________________ normalizin label too

data_frame = pd.read_csv('House Sales.csv')
data = data_frame.values[:, 2:]

#######   normalizing data   ###########
for i in range(19):
    data[:, i] = (data[:, i]-data[:, i].min())/(data[:, i].max()-data[:, i].min())
train_data = data[:4000,1:]
train_label = data[:4000, 0]
test_data = data[4000:5000,1:]
test_label = data[4000:5000, 0]




############  creating model   ############


m = 40  #number of first hidden layer's nuerons
k = 10
myModel = Sequential()
myModel.add(Dense(m, activation='relu', input_shape=(test_data[0, :].__len__(),)))
myModel.add(Dense(k, activation='relu'))
myModel.add(Dense(1, activation='relu'))

myModel.compile(optimizer=adagrad(.01), loss=mean_squared_error)


#########   training model   #########

network_history = myModel.fit(train_data, train_label, batch_size=60, epochs=10, validation_split=0.15) 
plot_(network_history)

##############   Evaluation   ##############
test_loss = myModel.evaluate(test_data, test_label)
test_label_peredicted = myModel.predict(test_data)
print(test_loss)