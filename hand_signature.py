# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:06:58 2020
@author: Ronald Tumuhairwe
"""
import keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

(mnist_train_images, mnist_train_labels),(mnist_test_images, mnist_test_labels) = mnist.load_data()

#convert the images (28x28) pixels to a 1-D array (1,784) for analysis
train_images = mnist_train_images.reshape(60000,784)
test_images = mnist_test_images.reshape(10000,784)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

#normalize the data
train_images /= 255
test_images /= 255

#convert the labels(integers) to binary_matrix
train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)

import matplotlib.pyplot as plt

def display_sample(num):
    #print the one-hot array of this sample label
    print(train_labels[num]) 
    #print the label converted back to a number
    label = train_labels[num].argmax(axis=0)
    image = train_images[num].reshape([28,28])
    plt.title('sample: %d Label: %d'% (num,label))
    plt.imshow(image,cmap=plt.get_cmap('gray_r'))
    plt.show() 

display_sample(1234)

model = Sequential()
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dense(10,activation='softmax'))
model.summary() 

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=100,epochs=10,verbose=2,validation_data=(test_images,test_labels))

score = model.evaluate(test_images,test_labels, verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
 
for x in range(1000):
    test_image = test_images[x,:].reshape(1,784)
    predicted_cat = model.predict(test_image).argmax()
    label = test_labels[x].argmax()
    if (predicted_cat != label):
        plt.title('prediction: %d Label: %d'%(predicted_cat,label))
        plt.imshow(test_image.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
        plt.show()
        
    
