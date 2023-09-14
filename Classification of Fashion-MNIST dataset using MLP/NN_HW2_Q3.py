import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import keras
from keras.losses import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import datetime
from keras import optimizers
    
###############     loading data     #########################
(train_data_img, train_label), (test_data_img, test_label) = fashion_mnist.load_data()


###############     showing firs 10 image     #################

for i in range(10):
    if train_label[i] == 0 :
        name = 'T-shirt'
    elif train_label[i] == 1 :
        name = 'Trouser'
    elif train_label[i] == 2 :
        name = 'Pullover'
    elif train_label[i] == 3 :
        name = 'Dress'
    elif train_label[i] == 4 :
        name = 'Coat'
    elif train_label[i] == 5 :
        name = 'Sandal'
    elif train_label[i] == 6 :
        name = 'Shirt'
    elif train_label[i] == 7 :
        name = 'Sneaker'
    elif train_label[i] == 8 :
        name = 'Bag'
    elif train_label[i] == 9 :
        name = 'Ankel boot'
    plt.subplot(1,10,i+1)
    imgplot = plt.imshow(train_data_img[i], cmap='gray')
    plt.axis('off')
    plt.title(name)
    plt.show()
    
##########################     preparing & normalizong data     ###################
train_data = train_data_img.reshape(60000, 784)
test_data = test_data_img.reshape(10000, 784)


 ### normalizing
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= np.max(train_data)
test_data /= np.max(test_data)

### making one-hot vector fot labels

train_label_1hot = np_utils.to_categorical(train_label)
test_label_1hot = np_utils.to_categorical(test_label)

#####################    creating model     ##############################


m = 1000              #number of neurons in firs hidden layer
k = 400            #number of neurons in second hidden layer
myModel = Sequential()
myModel.add(Dense(m, activation='relu',
                  input_shape=(train_data[0,:].__len__(), ))) 

myModel.add(keras.layers.Dense(k, activation='relu' ))

myModel.add(keras.layers.Dense(10, activation='softmax'))

myModel.compile(optimizer=optimizers.adam(0.0001),
                loss= categorical_crossentropy,
                metrics=['accuracy'])



##################     taining     ####################

start_training = datetime.datetime.now()
network_history = myModel.fit(train_data, train_label_1hot,
            batch_size=32, 
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


test_loss, test_acc = myModel.evaluate(test_data, test_label_1hot)
print('batch_size = ', 32)
print('Number of neurons in first hidden layer: ', m)
print('Number of neurons in second hidden layer: ', k)
print('Training Time: ', t)
print('test_loss: ', test_loss,'\t','test_acc: ', test_acc)
test_label_predicted_1hot = myModel.predict(test_data)
test_label_predicted = np.argmax(test_label_predicted_1hot, axis=1)

confusion_mtrx = confusion_matrix(test_label, test_label_predicted)









