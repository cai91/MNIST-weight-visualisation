import gzip
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense

def normIM(x):
    '''x: one dimensional vector e.g [1,2,3,4,5...n]'''
    x=np.array(x)
    x=(x-x.mean())/x.std()
    x=list(x)
    return x

for num_class in range(10):

    image_size = 28 # 28x28 pixel images
    num_images = 150

    # Reading digits
    f = gzip.open('train-images-idx3-ubyte.gz','r')
    f.read(16) # Offset for numbers 
    buf = f.read(image_size * image_size * num_images)
    X = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    X = X.reshape(num_images, image_size*image_size)
    X = [normIM(x) for x in X]

    # Reading labels
    f = gzip.open('train-labels-idx1-ubyte.gz','r')
    f.read(8) # Offset for labels 
    buf = f.read(num_images)
    y = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    y = [1 if y_i==num_class else 0 for y_i in y]

    # Setting the model
    model = Sequential()
    model.add(Dense(1, input_dim=784, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='RMSprop', metrics=['accuracy'])

    # Fit the model
    model.fit(X, y, epochs=250,class_weight='balanced',verbose=True)

    w=[i[0] for i in model.get_weights()[0]]+model.get_weights()[1]
    plt.imshow(w.reshape(1,28,28)[0],cmap='gray')
    plt.show()
