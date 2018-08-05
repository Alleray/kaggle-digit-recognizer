import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
from keras.backend import argmax
import os

#check if model already exists
if not os.path.isfile('model.h5'):
    print('No model.h5 found.')
    print('Loading training data...')
    train_data = np.genfromtxt('datasets/train.csv', delimiter=',', skip_header=1)

    print('Training data is loaded, got', train_data.shape[0], 'training examples. Shape:', train_data.shape)

    np.random.shuffle(train_data) #shuffling data before splitting to train and dev sets

    Y = train_data[:, 0]  #labels (0-9), 1st column of the train_data
    X = train_data[:, 1:] #pixel values, columns from 1 to end
    X /= 255 #normalizing parameters

    #splitting train_data to train and dev sets ~70-30%
    X_train = X[:30000]
    Y_train = Y[:30000]
    X_dev = X[30000:]
    Y_dev = Y[30000:]

    #convert integer Y to Y_one-hot vector
    Y_train_oh = to_categorical(Y_train, num_classes=10) 
    Y_dev_oh = to_categorical(Y_dev, num_classes=10)

    #print(Y.shape) #sanity checking
    #print(X.shape) #sanity checking

    def model(input_shape):
        X_input = Input(input_shape)

        X = Dense(1024, activation='relu')(X_input)
        X = Dropout(0.8)(X)

        X = Dense(512, activation='relu')(X)
        X = Dropout(0.8)(X)

        X = Dense(256, activation='relu')(X)
        X = Dropout(0.8)(X)

        X = Dense(10, activation='softmax')(X)

        model = Model(inputs=X_input, outputs=X)

        return model

    print('Initiating model...')
    model = model(X_train.shape[1:])
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

    print('Fitting model...')
    model.fit(x=X_train, y=Y_train_oh, epochs=40, batch_size=512)

    print('Fitting is complete.')
    print('Saving trained model...')
    model.save('model.h5')

    """
    #Uncomment this to evaluate model on the dev set
    print('Evaluating dev set...')
    predictions = model.evaluate(x=X_dev, y=Y_dev_oh)

    print()
    print('Loss =', round(predictions[0], 7))
    print('Dev set accuracy:', round(predictions[1], 3)
    """
else:
    print('Loading model...')
    model = load_model('model.h5')

print()
print('Loading test set...')
test_data = np.genfromtxt('datasets/test.csv', delimiter=',', skip_header=1)
test_data /= 255 #normalizing test data

print('Predicting labels...')
predictions = model.predict(x=test_data, verbose=1, batch_size=512)
predictions = np.argmax(predictions, axis=-1)
#print(predictions[0:10])

labels = predictions.astype(int) #getting rid of 2.000000000000000000e+00
imageid = np.arange(1, len(predictions)+1) #creating id axis

#adding new axis to be able to concatenate correctly
labels = labels[np.newaxis]
imageid = imageid[np.newaxis]

print(labels.shape, imageid.shape)

#concatenating and transposing
predictions = np.concatenate([imageid, labels]).T

#sanity check, format should be:
#ID - LABEL
#ID - LABEL
#ID - LABEL
#ID - LABEL
print(predictions[0:10])

np.savetxt('labels.csv', predictions, fmt='%i', delimiter=',', header='ImageId,Label', comments='')
print('Done!')