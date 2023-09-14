import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
import keras
from keras.losses import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import datetime
from keras import optimizers
from keras.datasets import fashion_mnist
from keras import layers 
# from keras import Model
from keras.models import Sequential
from sklearn.neural_network import BernoulliRBM
###############     loading data     #########################
(train_data_img, train_label), (test_data_img, test_label) = fashion_mnist.load_data()

##########################     preparing & normalizong data     ###################

train_data = train_data_img.reshape(train_data_img.__len__(), np.prod(train_data_img.shape[1:]))
test_data = test_data_img.reshape(test_data_img.__len__(), np.prod(train_data_img.shape[1:]))

  ### normalizing
 
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= np.max(train_data)
test_data /= np.max(test_data)


### making one-hot vector fot labels

train_label_1hot = np_utils.to_categorical(train_label)
test_label_1hot = np_utils.to_categorical(test_label)

####################     RBM      #########################

rbm = BernoulliRBM(n_components = 784, n_iter =20 ,learning_rate = 0.01,)

train_data_reduction = rbm.fit_transform(train_data)
test_data_reduction = rbm.fit_transform(test_data)



# train_data_reduction = train_data
# test_data_reduction = test_data


###################    creating model     ##############################

m = 1000              #number of neurons in firs hidden layer
k = 400            #number of neurons in second hidden layer
batch_size_ = 64
myModel = Sequential()
myModel.add(layers.Dense(m, activation='relu',
                  input_shape=(train_data_reduction[0,:].__len__(), ))) 

myModel.add(keras.layers.Dense(k, activation='relu' ))

myModel.add(keras.layers.Dense(10, activation='softmax'))

myModel.compile(optimizer=optimizers.adam(0.0001),
                loss= categorical_crossentropy,
                metrics=['accuracy'])



##################     taining     ####################

start_training = datetime.datetime.now()
network_history = myModel.fit(train_data_reduction, train_label_1hot,
            batch_size=batch_size_, 
            epochs=20, 
            validation_split=0.05)

end_training = datetime.datetime.now()
t = end_training - start_training 


dict_history = network_history.history
train_loss = dict_history['loss']
val_loss = dict_history['val_loss']
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['train', 'validation'])
plt.axis([-0.5,20, 0.01, 0.8])
plt.figure()

plt.title('Accuracy')
train_acc = dict_history['accuracy']
val_acc = dict_history['val_accuracy']
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(['train', 'validation'])
plt.axis([-0.5,20, 0.5, 1])
plt.show()


###################     test     #####################


test_loss, test_acc = myModel.evaluate(test_data_reduction, test_label_1hot)
print('batch_size = ', batch_size_)
print('Number of neurons in first hidden layer: ', m)
print('Number of neurons in second hidden layer: ', k)
print('Training Time: ', t)
print('test_loss: ', test_loss,'\t','test_acc: ', test_acc)
test_label_predicted_1hot = myModel.predict(test_data_reduction)
test_label_predicted = np.argmax(test_label_predicted_1hot, axis=1)

confusion_mtrx = confusion_matrix(test_label, test_label_predicted)












































