# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:15:48 2020

@author: khushi shah
"""
#importing the dataset
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

#initializing the cnn
classifier=Sequential()

#step1-convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#step2-maxpooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step3-flattening
classifier.add(Flatten())

#step4-fullconnection
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#compiling the cnn
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#part2-fitting the cnn to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Creating the Test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


classifier.fit_generator(training_set,samples_per_epoch=8000,nb_epoch=25,validation_data=test_set,nb_val_samples=2000)


#to predict new images 
def predict_image(imagepath, classifier):

    predict = image.load_img(imagepath, target_size = (64, 64))   

    predict_modified = image.img_to_array(predict)

    predict_modified = predict_modified / 255

    predict_modified = np.expand_dims(predict_modified, axis = 0)

    result = classifier.predict(predict_modified)

    if result[0][0] >= 0.5:

        prediction = 'dog'

        probability = result[0][0]

        print ("probability = " + str(probability))

    else:

        prediction = 'cat'

        probability = 1 - result[0][0]

        print ("probability = " + str(probability))

    print("Prediction = " + prediction)
predict_image('C:/Users/khushi shah/Downloads/catttt.jpg',classifier)