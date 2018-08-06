import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.backend import argmax
from keras.callbacks import ModelCheckpoint, TensorBoard
import os

#check if model already exists
if not os.path.isfile('model_cnn.h5'):
    print('Loading training data...')
    train_data = np.genfromtxt('datasets/train.csv', delimiter=',', skip_header=1)

    print('Training data is loaded, got', train_data.shape[0], 'training examples. Shape:', train_data.shape)

    np.random.shuffle(train_data)
    X = train_data[:, 1:] #pixel values, columns from 1 to end
    Y = train_data[:, 0] #labels (0-9), 1st column of the train_data
    X /= 255 #normalizing pixel values
    X = X.reshape(-1,28,28,1) #reshaping the training data to 28x28 matrix

    #splitting train_data to train and dev sets ~70-30%
    X_train = X[:30000]
    Y_train = Y[:30000]
    X_dev = X[30000:]
    Y_dev = Y[30000:]

    #convert integer Y to Y_one-hot vector
    Y_train_oh = to_categorical(Y_train, num_classes=10) 
    Y_dev_oh = to_categorical(Y_dev, num_classes=10)

    def model(input_shape):
        X_input = Input(input_shape)

        X = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same')(X_input)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = Dropout(0.8)(X)

        X = MaxPooling2D(2, 2)(X)
        X = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same')(X_input)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = Dropout(0.8)(X)

        X = Flatten()(X)

        X = Dense(200, activation='relu')(X)
        X = Dropout(0.8)(X)

        X = Dense(10, activation='softmax')(X)

        model = Model(inputs=X_input, outputs=X)

        return model

    print('Initiating model...')
    model = model(X_train.shape[1:])
    #adam = Adam(lr=0.0001, decay=0.00001)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    filepath = "checkpoints/model.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss',
                                verbose=1, save_best_only=True,
                                save_weights_only=False, mode='min')
    tensbrd = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    print('Fitting model...')
    model.fit(x=X_train, y=Y_train_oh, epochs=100, batch_size=512, callbacks=[checkpoint, tensbrd])

    print('Fitting is complete.')
    print('Saving trained model...')
    model.save('model_cnn.h5')

    #Uncomment this to evaluate model on the dev set
    print('Evaluating dev set...')
    predictions = model.evaluate(x=X_dev, y=Y_dev_oh)

    print()
    print('Loss =', round(predictions[0], 7))
    print('Dev set accuracy:', round(predictions[1], 3))

else:
    print('Loading model...')
    model = load_model('model_cnn.h5')

print()
print('Loading test set...')
test_data = np.genfromtxt('datasets/test.csv', delimiter=',', skip_header=1)
test_data /= 255 #normalizing test data
test_data = test_data.reshape(-1,28,28,1)


print('Predicting labels...')
predictions = model.predict(x=test_data, verbose=1, batch_size=512)
predictions = np.argmax(predictions, axis=-1)
#print(predictions[0:10])

labels = predictions.astype(int)
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
