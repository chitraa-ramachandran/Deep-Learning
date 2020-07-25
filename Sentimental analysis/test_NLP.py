#importing all necessary libraries
import numpy as np
import tensorflow as tf
import keras as k
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dense, Activation,Dropout,InputLayer,BatchNormalization,Embedding
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing import text
from tensorflow.keras import layers, models
from keras.preprocessing import sequence
import re
def remove(row):
    non_alpha = re.compile('[^a-zA-Z]')
    spcl_char = re.compile('[;.!:\',\"()?\[\]]')
    html = re.compile(r'<.*?>')
    row= [html.sub(" ",i) for i in row]
    row = [spcl_char.sub(" ", i) for i in row]
    row = [non_alpha.sub(" ", i.lower()) for i in row]
    return row
if __name__ == "__main__":
    # creating the test dataset with positive and negative reviews
    neg_test = []
    for root, directory, files in os.walk("data/aclImdb/test/neg", topdown=True):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    data = f.read()
                    neg_test.append(data)
    pos_test = []
    for root, directory, files in os.walk("data/aclImdb/test/pos", topdown=True):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    data = f.read()
                    pos_test.append(data)
    polarity = [0 if i < 12500 else 1 for i in range(25000)]
    # creating the train and test set by appending together both the positive and negative reviews
    #print("before",len(neg_test))
    for i in range(12500):
        neg_test.append(pos_test[i])
    #print(len(neg_test))
    X_test, y_test = ([] for i in range(2))
    for i in list(range(25000)):
        X_test.append(neg_test[i])
        y_test.append(polarity[i])
    X_test = remove(X_test)
    # loading tokenizer object
    pickle_obj = open("models/tokenizerobj.pickle", "rb")
    tokenizer = pickle.load(pickle_obj)
    max_length = 500
    vocabulary = 5000
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = sequence.pad_sequences(X_test, maxlen=max_length)
    y_test = [0 if i < 12500 else 1 for i in range(25000)]
    y_test = np.asarray(y_test)
    imdb1 = models.Sequential()
    imdb1.add(layers.Embedding(vocabulary, 64, input_length=max_length))
    imdb1.add(layers.Dropout(0.4))
    imdb1.add(layers.Flatten())
    imdb1.add(layers.Dense(1, activation='sigmoid'))
    op = tf.keras.optimizers.Adam()
    imdb1.compile(optimizer=op, loss='binary_crossentropy', metrics=['accuracy'])
    imdb1.load_weights("models/20820331_NLP_model.h5")
    loss, accuracy = imdb1.evaluate(X_test, y_test)
    print("Accuracy on test data : ", accuracy)