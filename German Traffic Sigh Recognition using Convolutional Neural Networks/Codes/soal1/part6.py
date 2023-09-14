import numpy as np
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM, Dropout, GRU
import tensorflow as tf

def load_data(n):  #random time in week
    all_data = np.load('polution_dataSet.npy')
    p = list(range(n +168*3,43799,168*4))   # labels
    q = [m-i for m in p for i in range(11,0,-1) ]   
    X_train = np.copy(all_data[tuple(q), :])
    Y_train = np.copy(all_data[tuple(p), 0])
    X_train = np.reshape(X_train, (int(X_train.shape[0]/11), 11, X_train.shape[1]))
    return X_train, Y_train
def load_data_weekly_seris(n):  #random time in week
    all_data = np.load('polution_dataSet.npy')
    p = list(range(n +168*3,43799,168*4))   # labels
    q = [m-i for m in p for i in [144,120,96,72,48,24]]
    X_train = np.copy(all_data[tuple(q), :])
    Y_train = np.copy(all_data[tuple(p), 0])
    X_train = np.reshape(X_train, (int(X_train.shape[0]/6), 6, X_train.shape[1]))
    return X_train, Y_train   
def load_data_monthly_seris(n) :
    all_data = np.load('polution_dataSet.npy')
    a = list(range(0,260,1))              ### all rows
    b = list(range(3,260,4))              ### laybel rows
    c = [x for x in a if x not in b]      ### rows except labels
    all_data_monthly = np.copy(all_data[tuple(range(n,43799,168)), :])
    X = np.copy(all_data_monthly[tuple(c), :])
    Y = np.copy(all_data_monthly[tuple(b), :])
    X_train = np.copy(X[:int(X.__len__()), :])
    Y_train = np.copy(Y[:int(Y.__len__()), 0])
    X_train = np.reshape(X_train, (int(X_train.shape[0]/3), 3, X_train.shape[1]))
    return X_train, Y_train


n = random.randint(1,168) - 1   # random hour
x_train_month, Y_train_month = load_data_monthly_seris(n)
x_train_week, Y_train_week = load_data_weekly_seris(n)
x_train_day, y_train_day = load_data(n)



input_day = tf.keras.layers.Input(shape=(11,8))
lstm1 = tf.keras.layers.GRU(50)(input_day)
x1 = tf.keras.layers.Dense(1, activation='relu')(lstm1)


input_week = tf.keras.layers.Input(shape=(6,8))
lstm2 = tf.keras.layers.GRU(50)(input_week)
x2 = tf.keras.layers.Dense(1, activation='relu')(lstm2)

input_month = tf.keras.layers.Input(shape=(3,8))
lstm3 = tf.keras.layers.GRU(50)(input_month)
x3 = tf.keras.layers.Dense(1, activation='relu')(lstm3)

avg = tf.keras.layers.Average()([x1, x2, x3])
out = tf.keras.layers.Dense(1)(avg)
model = tf.keras.models.Model(inputs=[input_day, input_week,input_month], outputs=out)

model.compile(loss='mse', optimizer='adam')
model.summary()

history = model.fit([x_train_day, x_train_week, x_train_month]
                    ,[y_train_day] ,epochs=100,batch_size=12,
                    validation_split=0.15)

l_val = history.history['val_loss']
l_tr = history.history['loss']
plt.plot(l_tr)
plt.plot(l_val)
plt.legend(['train', 'validation'])
plt.show()













