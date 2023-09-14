import numpy as np
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM, Dropout, GRU
import time
def load_data():
    all_data = np.load('polution_dataSet.npy')
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

def lstm_feature_selection(epoch): 
    X_train, Y_train, X_test, Y_test = load_data()
    cnt = 0
    err = []
    t1 = time.time()
    for i in range(1, 7) :
        for j in range(i+1,8) :
            X_train_s = np.copy(X_train[:, (0, i, j)])
            X_test_s = np.copy(X_test[:, (0, i, j)])
            X_train_s = np.reshape(X_train_s, (int(X_train_s.shape[0]/11), 11, X_train_s.shape[1]))
            X_test_s  = np.reshape(X_test_s , (int(X_test_s.shape[0]/11), 11, X_test_s.shape[1]))
            los = 'mean_squared_error'
            opt = 'adam'
            cnt+=1
            model = Sequential()
            model.add(LSTM(50, input_shape=(11, 3)))
            model.add(Dense(1))
            model.compile(loss=los, optimizer=opt)
            history = model.fit(X_train_s, Y_train,
                                epochs=epoch, batch_size=12, verbose=2,
                                validation_data=(X_test_s, Y_test))
            plt.figure('loss', figsize=(20,10))
            plt.subplot(6,4,cnt)
            plt.tight_layout()
            plt.title('feature 0 ,'+str(i)+','+str(j)+ 'have been selected')
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.xticks([])
            if cnt ==1 :
                plt.legend(['train_loss', 'test_loss'])
            y_predicted = model.predict(X_test_s)
            plt.figure('predict&actual', figsize=(20,10))
            plt.subplot(6,4,cnt)
            plt.tight_layout()
            plt.title('feature 0 ,'+str(i)+','+str(j)+ 'have been selected')
            plt.plot(y_predicted)
            plt.plot(Y_test)
            plt.axis('off')
            if cnt ==1 :
                plt.legend(['predicted', 'actual'])
            error = math.sqrt(mean_squared_error(Y_test, y_predicted))
            err.append(error)
    t2 = time.time()
    duration = t2 - t1
    return err, duration
def rnn_feature_selection(epoch): 
    X_train, Y_train, X_test, Y_test = load_data()
    cnt = 0
    err = []
    t1 = time.time()
    for i in range(1, 7) :
        for j in range(i+1,8) :
            X_train_s = np.copy(X_train[:, (0, i, j)])
            X_test_s = np.copy(X_test[:, (0, i, j)])
            X_train_s = np.reshape(X_train_s, (int(X_train_s.shape[0]/11), 11, X_train_s.shape[1]))
            X_test_s  = np.reshape(X_test_s , (int(X_test_s.shape[0]/11), 11, X_test_s.shape[1]))
            los = 'mean_squared_error'
            opt = 'adam'
            cnt+=1
            model = Sequential()
            model.add(SimpleRNN(units = 50, input_shape=(11, 3)))
            model.add(Dense(1))
            model.compile(loss=los, optimizer=opt)
            history = model.fit(X_train_s, Y_train,
                                epochs=epoch, batch_size=12, verbose=2,
                                validation_data=(X_test_s, Y_test))
            plt.figure('loss', figsize=(20,10))
            plt.subplot(6,4,cnt)
            plt.tight_layout()
            plt.title('feature 0 ,'+str(i)+','+str(j)+ 'have been selected')
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.xticks([])
            if cnt ==1 :
                plt.legend(['train_loss', 'test_loss'])
            y_predicted = model.predict(X_test_s)
            plt.figure('predict&actual', figsize=(20,10))
            plt.subplot(6,4,cnt)
            plt.tight_layout()
            plt.title('feature 0 ,'+str(i)+','+str(j)+ 'have been selected')
            plt.plot(y_predicted)
            plt.plot(Y_test)
            plt.axis('off')
            if cnt ==1 :
                plt.legend(['predicted', 'actual'])
            error = math.sqrt(mean_squared_error(Y_test, y_predicted))
            err.append(error)
    t2 = time.time()
    duration = t2 - t1
    return err, duration
def gru_feature_selection(epoch): 
    X_train, Y_train, X_test, Y_test = load_data()
    cnt = 0
    err = []
    t1 = time.time()
    for i in range(1, 7) :
        for j in range(i+1,8) :
            X_train_s = np.copy(X_train[:, (0, i, j)])
            X_test_s = np.copy(X_test[:, (0, i, j)])
            X_train_s = np.reshape(X_train_s, (int(X_train_s.shape[0]/11), 11, X_train_s.shape[1]))
            X_test_s  = np.reshape(X_test_s , (int(X_test_s.shape[0]/11), 11, X_test_s.shape[1]))
            los = 'mean_squared_error'
            opt = 'adam'
            cnt+=1
            model = Sequential()
            model.add(GRU(units = 50, input_shape=(11, 3)))
            model.add(Dense(1))
            model.compile(loss=los, optimizer=opt)
            history = model.fit(X_train_s, Y_train,
                                epochs=epoch, batch_size=12, verbose=2,
                                validation_data=(X_test_s, Y_test))
            plt.figure('loss', figsize=(20,10))
            plt.subplot(6,4,cnt)
            plt.tight_layout()
            plt.title('feature 0 ,'+str(i)+','+str(j)+ 'have been selected')
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.xticks([])
            if cnt ==1 :
                plt.legend(['train_loss', 'test_loss'])
            y_predicted = model.predict(X_test_s)
            plt.figure('predict&actual', figsize=(20,10))
            plt.subplot(6,4,cnt)
            plt.tight_layout()
            plt.title('feature 0 ,'+str(i)+','+str(j)+ 'have been selected')
            plt.plot(y_predicted)
            plt.plot(Y_test)
            plt.axis('off')
            if cnt ==1 :
                plt.legend(['predicted', 'actual'])
            error = math.sqrt(mean_squared_error(Y_test, y_predicted))
            err.append(error)
    t2 = time.time()
    duration = t2 - t1
    return err, duration

####################################################################
def lstm_3_best_features(epoch, i,j):   #i, j is the 2nd and 3rd features
    X_train, Y_train, X_test, Y_test = load_data()
    X_train_s = np.copy(X_train[:, (0, i, j)])
    X_test_s = np.copy(X_test[:, (0, i, j)])
    X_train_s = np.reshape(X_train_s, (int(X_train_s.shape[0]/11), 11, X_train_s.shape[1]))
    X_test_s  = np.reshape(X_test_s , (int(X_test_s.shape[0]/11), 11, X_test_s.shape[1]))
    los = 'mean_squared_error'
    opt = 'adam'
    model = Sequential()
    model.add(LSTM(50, input_shape=(11, 3)))
    model.add(Dense(1))
    model.compile(loss=los, optimizer=opt)
    history = model.fit(X_train_s, Y_train,
                        epochs=epoch, batch_size=12, verbose=2,
                        validation_data=(X_test_s, Y_test))
    plt.figure('loss')
    plt.title('LSTM; feature 0 ,'+str(i)+','+str(j)+ 'have been selected')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train_loss', 'test_loss'])
    y_predicted = model.predict(X_test_s)
    plt.figure('predict&actual')
    plt.title('LSTM; feature 0 ,'+str(i)+','+str(j)+ 'have been selected')
    plt.plot(y_predicted)
    plt.plot(Y_test)
#    plt.axis('off')
    plt.legend(['predicted', 'actual'])
    error = math.sqrt(mean_squared_error(Y_test, y_predicted))
    print(error)
    return error

def rnn_3_best_features(epoch, i,j):   #i, j is the 2nd and 3rd features
    X_train, Y_train, X_test, Y_test = load_data()
    X_train_s = np.copy(X_train[:, (0, i, j)])
    X_test_s = np.copy(X_test[:, (0, i, j)])
    X_train_s = np.reshape(X_train_s, (int(X_train_s.shape[0]/11), 11, X_train_s.shape[1]))
    X_test_s  = np.reshape(X_test_s , (int(X_test_s.shape[0]/11), 11, X_test_s.shape[1]))
    los = 'mean_squared_error'
    opt = 'adam'
    model = Sequential()
    model.add(SimpleRNN(units = 50, input_shape=(11,3)))
    model.add(Dense(1))
    model.compile(loss=los, optimizer=opt)
    history = model.fit(X_train_s, Y_train,
                        epochs=epoch, batch_size=12, verbose=2,
                        validation_data=(X_test_s, Y_test))
    plt.figure('loss')
    plt.title('RNN; feature 0 ,'+str(i)+','+str(j)+ 'have been selected')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train_loss', 'test_loss'])
    y_predicted = model.predict(X_test_s)
    plt.figure('predict&actual')
    plt.title('RNN; feature 0 ,'+str(i)+','+str(j)+ 'have been selected')
    plt.plot(y_predicted)
    plt.plot(Y_test)
#    plt.axis('off')
    plt.legend(['predicted', 'actual'])
    error = math.sqrt(mean_squared_error(Y_test, y_predicted))
    print(error)
    return error

def gru_3_best_features(epoch, i,j):   #i, j is the 2nd and 3rd features
    X_train, Y_train, X_test, Y_test = load_data()
    X_train_s = np.copy(X_train[:, (0, i, j)])
    X_test_s = np.copy(X_test[:, (0, i, j)])
    X_train_s = np.reshape(X_train_s, (int(X_train_s.shape[0]/11), 11, X_train_s.shape[1]))
    X_test_s  = np.reshape(X_test_s , (int(X_test_s.shape[0]/11), 11, X_test_s.shape[1]))
    los = 'mean_squared_error'
    opt = 'adam'
    model = Sequential()
    model.add(GRU(units = 50, input_shape=(11, 3)))
    model.add(Dense(1))
    model.compile(loss=los, optimizer=opt)
    history = model.fit(X_train_s, Y_train,
                        epochs=epoch, batch_size=12, verbose=2,
                        validation_data=(X_test_s, Y_test))
    plt.figure('loss')
    plt.title('GRU; feature 0 ,'+str(i)+','+str(j)+ 'have been selected')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train_loss', 'test_loss'])
    y_predicted = model.predict(X_test_s)
    plt.figure('predict&actual')
    plt.title('GRU; feature 0 ,'+str(i)+','+str(j)+ 'have been selected')
    plt.plot(y_predicted)
    plt.plot(Y_test)
#    plt.axis('off')
    plt.legend(['predicted', 'actual'])
    error = math.sqrt(mean_squared_error(Y_test, y_predicted))
    print(error)
    return error
part = input('feature selection / network with 3 feature?(1/2): ')

net = input(' Tell me which network( lstm / rnn / gru)? :' )
if part =='1' :
    manner = input('which manner(1.exhustive search / 2.pca) ?:')
    if manner == '1' : 
        if net == 'lstm' :
            error_lstm_s, duration_lstm = lstm_feature_selection(100)
        elif net == 'rnn' :
            error_rnn_s, duration_rnn = rnn_feature_selection(100)
        elif net == 'gru' :
            error_gru_s, duration_gru = gru_feature_selection(100)
    elif manner == '2' :
        X_train, Y_train, X_test, Y_test = load_data()
        covariance = np.cov(X_train.T)
        feature_variance, _ = np.linalg.eig(covariance)
        print(feature_variance)        

if part =='2' :
    if net == 'lstm' :
        error_lstm = lstm_3_best_features(100,1 ,2)
    elif net == 'rnn' :
       error_rnn = rnn_3_best_features(100, 1, 2)
    elif net == 'gru' :
        error_gru = gru_3_best_features(100, 1, 2)
    
1
