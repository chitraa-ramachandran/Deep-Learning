# import necessary packages
import numpy as np
import keras as k
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dense, Activation,Dropout,InputLayer,LSTM, Embedding,GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import savetxt
from numpy import loadtxt
import pickle
if __name__ == "__main__":
#loading the saved model
    stockGRU = tf.keras.models.load_model('models/20820331_RNN_model/')
    # Load the test data and seperate the feature and the target arrays
    Test_Data = loadtxt('data/test_data_RNN.csv', delimiter=',')
    x_test = Test_Data[:, :12]
    y_test = Test_Data[:, 12:]
    #load the scaler pickel file inorder to fit transform the xtest data to be fed as a input to the model for prediction
    #open the pickle file in read mode
    input_scaler = open("models/scalerobj.pickle", "rb")
    #load the scaler object
    scaler = pickle.load(input_scaler)
    #transforming the test data according to the scaler object
    x_test = scaler.transform(x_test)
    x_test = x_test.astype('float32')
    #convert x_test as a 3 dimensional data that will be accepted as the input shape to the lstm layers used in our model
    x_test = np.reshape(x_test, (x_test.shape[0], 3, -1))
    #predict the target value through our trained model
    pred = stockGRU.predict(x_test)
    loss, accuracy = stockGRU.evaluate(x_test,y_test)
    print("Results ",loss," ",accuracy)
    #plotting the obtained graph
    plt.xlabel('Days')
    plt.ylabel('Price of the stock')
    plt.plot(y_test,color='red',label='Actual target vale')
    plt.plot(pred,color='green',label='Obtained target value')
    plt.legend()
    plt.show()
