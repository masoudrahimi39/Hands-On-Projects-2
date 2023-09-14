import numpy as np
import random
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM, Dropout, GRU
def load_data_weekly_seris() :
    n = random.randint(1,24) - 1   # random hour
    all_data = np.load('polution_dataSet.npy')
    a = list(range(0,1820,1))              ### all rows
    b = list(range(6,1820,7))              ### laybel rows
    c = [x for x in a if x not in b]      ### rows except labels
    all_data_weekly = np.copy(all_data[tuple(range(n,43799,24)), :])
    X = np.copy(all_data_weekly[tuple(c), :])
    Y = np.copy(all_data_weekly[tuple(b), :])
    X_train = np.copy(X[:int(X.__len__()*0.8), :])
    Y_train = np.copy(Y[:int(Y.__len__()*0.8), 0])
    X_test = np.copy(X[int(X.__len__()*0.8):, :])
    Y_test = np.copy(Y[int(Y.__len__()*0.8):, 0])
    return X_train, Y_train, X_test, Y_test


def load_data_monthly_seris() :
    n = random.randint(1,168) - 1   # random hour
    all_data = np.load('polution_dataSet.npy')
    a = list(range(0,260,1))              ### all rows
    b = list(range(3,260,4))              ### laybel rows
    c = [x for x in a if x not in b]      ### rows except labels
    all_data_monthly = np.copy(all_data[tuple(range(n,43799,168)), :])
    X = np.copy(all_data_monthly[tuple(c), :])
    Y = np.copy(all_data_monthly[tuple(b), :])
    X_train = np.copy(X[:int(X.__len__()*0.8), :])
    Y_train = np.copy(Y[:int(Y.__len__()*0.8), 0])
    X_test = np.copy(X[int(X.__len__()*0.8):, :])
    Y_test = np.copy(Y[int(Y.__len__()*0.8):, 0])
    return X_train, Y_train, X_test, Y_test

def lstm_part4_weekly(epoch): 
    X_train, Y_train, X_test, Y_test  = load_data_weekly_seris()
    X_train = np.reshape(X_train, (int(X_train.shape[0]/6), 6, X_train.shape[1]))
    X_test = np.reshape(X_test, (int(X_test.shape[0]/6), 6, X_test.shape[1]))
    model = Sequential()
    model.add(LSTM(50, input_shape=(6, 8)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    begin = time.time()
    history = model.fit(X_train, Y_train,
                        epochs=epoch, batch_size=12, verbose=2,
                        validation_data=(X_test, Y_test))
    end = time.time()
    print('learning time is:%0.2f'%(end - begin))
    y_predicted = model.predict(X_test)
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    error = sqrt(mean_squared_error(Y_test, y_predicted))
    print('RMSE for test data is : %.3f' % error)

    plt.figure('loss')
    plt.title('LSTM; for weekly serie')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train_loss', 'test_loss'])
    plt.figure('predict&actual')
    plt.title('LSTM; for weekly serie')
    plt.plot(y_predicted)
    plt.plot(Y_test)
    plt.legend(['predicted', 'actual'])
import time
def rnn_part4_weekly(epoch):   
    X_train, Y_train, X_test, Y_test = load_data_weekly_seris()
    X_train = np.reshape(X_train, (int(X_train.shape[0]/6), 6, X_train.shape[1]))
    X_test = np.reshape(X_test, (int(X_test.shape[0]/6), 6, X_test.shape[1]))
    los = 'mean_squared_error'
    opt = 'adam'
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(6, 8)))
    model.add(Dense(1))
    model.compile(loss=los, optimizer=opt)
    begin = time.time()
    history = model.fit(X_train, Y_train,
                        epochs=epoch, batch_size=12, verbose=2,
                        validation_data=(X_test, Y_test))
    end = time.time()
    print('learning time is:%0.2f'%(end - begin))
    plt.figure('loss')
    plt.title('RNN; for weekly serie')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train_loss', 'test_loss'])
    y_predicted = model.predict(X_test)
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    error = sqrt(mean_squared_error(Y_test, y_predicted))
    print('RMSE for test data is : %.3f' % error)
    plt.figure('predict&actual')
    plt.title('RNN; for weekly serie')
    plt.plot(y_predicted)
    plt.plot(Y_test)
    plt.legend(['predicted', 'actual'])
#rnn_part4_weekly(100)
def gru_part4_weekly(epoch):   
    X_train, Y_train, X_test, Y_test = load_data_weekly_seris()
    X_train = np.reshape(X_train, (int(X_train.shape[0]/6), 6, X_train.shape[1]))
    X_test = np.reshape(X_test, (int(X_test.shape[0]/6), 6, X_test.shape[1]))
    los = 'mean_squared_error'
    opt = 'adam'
    model = Sequential()
    model.add(GRU(50, input_shape=(6, 8)))
    model.add(Dense(1))
    model.compile(loss=los, optimizer=opt)
    begin = time.time()
    history = model.fit(X_train, Y_train,
                        epochs=epoch, batch_size=12, verbose=2,
                        validation_data=(X_test, Y_test))
    end = time.time()
    print(' learning time is:%0.2f'%(end - begin))
    plt.figure('loss')
    plt.title('GRU; for weekly serie')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train_loss', 'test_loss'])
    y_predicted = model.predict(X_test)
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    error = sqrt(mean_squared_error(Y_test, y_predicted))
    print('RMSE for test data is : %.3f' % error)
    plt.figure('predict&actual')
    plt.title('GRU; for weekly serie')
    plt.plot(y_predicted)
    plt.plot(Y_test)
    plt.legend(['predicted', 'actual'])
    
def lstm_part4_monthly(epoch): 
    X_train, Y_train, X_test, Y_test  = load_data_monthly_seris()
    X_train = np.reshape(X_train, (int(X_train.shape[0]/3), 3, X_train.shape[1]))
    X_test = np.reshape(X_test, (int(X_test.shape[0]/3), 3, X_test.shape[1]))
    model = Sequential()
    model.add(LSTM(50, input_shape=(3, 8)))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    begin = time.time()
    history = model.fit(X_train, Y_train,
                        epochs=epoch, batch_size=22, verbose=2,
                        validation_data=(X_test, Y_test))
    end = time.time()
    print(' learning time is:%0.2f'%(end - begin))
    plt.figure('loss')
    plt.title('LSTM; for monthly serie')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train_loss', 'test_loss'])
    y_predicted = model.predict(X_test)
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    error = sqrt(mean_squared_error(Y_test, y_predicted))
    print('RMSE for test data is : %.3f' % error)
    plt.figure('predict&actual')
    plt.title('LSTM; for monthly serie')
    plt.plot(y_predicted)
    plt.plot(Y_test)
    plt.legend(['predicted', 'actual'])

def rnn_part4_monthly(epoch):   
    X_train, Y_train, X_test, Y_test = load_data_monthly_seris()
    X_train = np.reshape(X_train, (int(X_train.shape[0]/3), 3, X_train.shape[1]))
    X_test = np.reshape(X_test, (int(X_test.shape[0]/3), 3, X_test.shape[1]))
    los = 'mean_squared_error'
    opt = 'adam'
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(3, 8)))
    model.add(Dense(1))
    model.compile(loss=los, optimizer=opt)
    begin = time.time()
    history = model.fit(X_train, Y_train,
                        epochs=epoch, batch_size=22, verbose=2,
                        validation_data=(X_test, Y_test))
    end = time.time()
    print('learning time is:%0.2f'%(end - begin))
    plt.figure('loss')
    plt.title('RNN; for monthly serie')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train_loss', 'test_loss'])
    y_predicted = model.predict(X_test)
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    error = sqrt(mean_squared_error(Y_test, y_predicted))
    print('RMSE for test data is : %.3f' % error)
    plt.figure('predict&actual')
    plt.title('RNN; for monthly serie')
    plt.plot(y_predicted)
    plt.plot(Y_test)
    plt.legend(['predicted', 'actual'])

def gru_part4_monthly(epoch):   
    X_train, Y_train, X_test, Y_test = load_data_monthly_seris()
    X_train = np.reshape(X_train, (int(X_train.shape[0]/3), 3, X_train.shape[1]))
    X_test = np.reshape(X_test, (int(X_test.shape[0]/3), 3, X_test.shape[1]))
    los = 'mean_squared_error'
    opt = 'adam'
    model = Sequential()
    model.add(GRU(50, input_shape=(3, 8)))
    model.add(Dense(1))
    model.compile(loss=los, optimizer=opt)
    begin = time.time()
    history = model.fit(X_train, Y_train,
                        epochs=epoch, batch_size=22, verbose=2,
                        validation_data=(X_test, Y_test))
    end = time.time()
    print('learning time is:%0.2f'%(end - begin))
    plt.figure('loss')
    plt.title('GRU; for monthly serie')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train_loss', 'test_loss'])
    y_predicted = model.predict(X_test)
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    error = sqrt(mean_squared_error(Y_test, y_predicted))
    print('RMSE for test data is : %.3f' % error)
    plt.figure('predict&actual')
    plt.title('GRU; for monthly serie')
    plt.plot(y_predicted)
    plt.plot(Y_test)
    plt.legend(['predicted', 'actual'])
    
    
    
    
    
#lstm_part4_weekly(100)
#rnn_part4_weekly(100)
#gru_part4_weekly(100)
#lstm_part4_monthly(100)
#rnn_part4_monthly(100)
gru_part4_monthly(5)



