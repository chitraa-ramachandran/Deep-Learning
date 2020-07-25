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
import pickle


def remove(row):
    non_alpha = re.compile('[^a-zA-Z]')
    spcl_char = re.compile('[;.!:\',\"()?\[\]]')
    html = re.compile(r'<.*?>')
    row= [html.sub(" ",i) for i in row]
    row = [spcl_char.sub(" ", i) for i in row]
    row = [non_alpha.sub(" ", i.lower()) for i in row]
    return row
if __name__ == "__main__":
    # creating the training dataset with positive and negative reviews
    """
    The Large Movie Review Dataset contains 25000 highly polar reviews for training and 25000 dataset for testing. 
    So, we are attempting to do a binary classification on the reviews by classifying them as positive or negative 
    reviews by using natural language processing.

    In the downloaded datafile the ‘pos’ and the ‘neg’ files contain the positive and the negative reviews respectively. 
    Considering these two files as input we need to create our data that will have the positive and negative reviews 
    with labels 1 and 0 for good and bad respectively.
    """
    neg = []
    for root, directory, files in os.walk("data/aclImdb/train/neg", topdown=True):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    data = f.read()
                    neg.append(data)
    pos = []
    for root, directory, files in os.walk("data/aclImdb/train/pos", topdown=True):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    data = f.read()
                    pos.append(data)
    # creating the target polarity label with 1 for positive reviews and 0 for negative reviews
    polarity = [0 if i < 12500 else 1 for i in range(25000)]
    # creating the train and test set by appending together both the positive and negative reviews
    for i in range(12500):
        neg.append(pos[i])
    X_train, y_train  = ([] for i in range(2))
    for i in list(range(25000)):
        X_train.append(neg[i])
        y_train.append(polarity[i])
    # #plotting an histogram for length of the reviews
    # plt.figure(figsize=(10,10))
    # plt.hist([len(review) for review in list(X_train)], 50)
    # plt.xlabel('Length')
    # plt.ylabel('Frequency')
    # plt.title('Distribution Graph')
    # plt.show()
    """
    The reviews contain a lot of unwanted data that do not hold any information to train the model and 
    they need to be removed by doing data preprocessing.The unwanted data observed in the reviews are numbers,
    punctuations,html tags and other special characters.These can be identified and removed using the regex matches 
    and substitutions.
    """
    # data preprocessing
    # removing the unwanted data from the train and test
    X_train = remove(X_train)
    # process of tokenization
    """
    Tokenization is the process of splitting a text into a list of tokens thereby creating a word-to-index dictionary 
    where each word is considered as a key and a unique index is created for it. 
    """
    vocabulary = 5000
    tokenizer = Tokenizer(num_words=vocabulary)
    # we create the indexes based on word frequency so,we use fit_on_texts funtion from keras tokenizer
    tokenizer.fit_on_texts(X_train)
    # transforms each text in texts to a sequence of integer using texts_to_sequence to form a valid input to the neural network
    X_train = tokenizer.texts_to_sequences(X_train)
    # creating a pickle file
    filename = "models/tokenizerobj.pickle"
    # open the pickle file
    filehandler = open(filename, "wb")
    # dump the pickle file by loading the scaler
    pickle.dump(tokenizer, filehandler)
    # close the pickle file
    filehandler.close()
    '''
    X_train and X_test are now a list of values and each list corresponds to the sentence in the train and test set.
    For purpose of training (To provide a proper input to the neural network) we set a max length to the list in such a way that if the size of the list is greater than 
    the length mentioned then we truncate it and in the other scenario when the size is small we pad the list with 0
    '''
    max_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_length)

    # load the training dataset and create labels
    y_train = [0 if i < 12500 else 1 for i in range(25000)]
    y_train = np.asarray(y_train)
    # Splitting the training dataset into train and validation data to analyse the performance of the model
    x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # Fixing the vocabulary and maximum length for the review
    vocabulary = 5000
    max_length = 500
    tf.keras.backend.clear_session()
    # defining the model
    imdb = models.Sequential()
    # adding the embedding layer which takes the input length and vocabulary size
    imdb.add(Embedding(vocabulary, 64, input_length=max_length))
    imdb.add(Dropout(0.2))
    imdb.add(Flatten())
    imdb.add(Dense(1, activation='sigmoid'))
    # compiling the model using adam optimizer and binary_crossentropy
    op = tf.keras.optimizers.Adam()
    imdb.compile(optimizer=op, loss='binary_crossentropy', metrics=['accuracy'])
    imdb.summary()
    # store the trained model
    file_name = "models/20820331_NLP_model.h5"
    checkpoint = k.callbacks.ModelCheckpoint(file_name,
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True,
                                             mode='min')
    history = imdb.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=64, verbose=2,
                       callbacks=[checkpoint])

    # plot the training and validation accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    # plot the training and validation loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    loss, accuracy = imdb.evaluate(x_train, y_train)
    print("Accuracy on train: ", accuracy, "Loss on train: ", loss)
    loss, accuracy = imdb.evaluate(x_val, y_val)
    print("Accuracy on val: ", accuracy, "Loss on val: ", loss)
