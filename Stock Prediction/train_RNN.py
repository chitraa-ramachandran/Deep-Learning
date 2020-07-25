# import necessary packages
import numpy as np
import keras as k
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dense, Activation,Dropout,InputLayer,LSTM, Embedding,GRU,SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import savetxt
from numpy import loadtxt
import pickle
#function to create the train and test RNN csv file which needs to be commented out
def Data_Creation():
    data_org = pd.read_csv("data/q2_dataset.csv")
    data_org=data_org.iloc[:, 2:]
    data_org = data_org.to_numpy()
    custom_data=[]
    for i in range(len(data_org)-3):
        temp=[]
        for j in range(3):
            temp.append(data_org[i+j][0])
            temp.append(data_org[i+j][1])
            temp.append(data_org[i+j][2])
            temp.append(data_org[i+j][3])
        temp.append(data_org[i+3][1])
        custom_data.append(temp)
    custom_data=np.asarray(custom_data)
    #randomizing the data
    random_index=np.arange(len(custom_data))
    np.random.shuffle(random_index)
    custom_data=custom_data[random_index]
    #spliting the created dataset in the ratio of 7:3
    split_len=int(len(custom_data)*0.7)
    Train=custom_data[:split_len]
    Test=custom_data[split_len:]
    #writing the test and the train data into the respective csv files
    savetxt('data/train_data_RNN.csv', Train, delimiter=',')
    savetxt('data/test_data_RNN.csv', Test, delimiter=',')
if __name__ == "__main__":
    #calling the data creation function
    # Data_Creation()
    #load the training dataset
    Train_Data = loadtxt('data/train_data_RNN.csv', delimiter=',')
    #normalizing the data using the filename = "models/scalerobj.pickle"ues lie between the feature range of 0,1
    X_Train = Train_Data[:, :12]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_Train = scaler.fit_transform(X_Train)
    #creating a pickle file
    filename="models/scalerobj.pickle"
    #open the pickle file
    filehandler = open(filename, "wb")
    #dump the pickle file by loading the scaler
    pickle.dump(scaler, filehandler)
    #close the pickle file
    filehandler.close()
    yTrain = Train_Data[:,12]
    """
    FOR STOCK PREDICTION we will be using the LSTM,GRU,RNN layers which accepts the input in three dimension and hence we need to reshape     the x_train data to 3 dimensional input
    """
    X_Train = np.reshape(X_Train, (X_Train.shape[0], 3, -1))
    # creating a validation dataset from the train data by spliting the original train data in the ratio of 8:2
    X_Train,X_val,y_train,y_val = train_test_split(X_Train,yTrain,test_size=0.2)
    #creating the training model
    stockGRU=Sequential()
    stockGRU.add(GRU(units=100, return_sequences=True,input_shape=(X_Train.shape[1], X_Train.shape[2])))
    stockGRU.add(GRU(units=50, return_sequences=True))
    stockGRU.add(GRU(units=50))
    stockGRU.add(Dense(units=1))
    #compiling the model
    stockGRU.compile(optimizer='adam',loss='mean_squared_error',metrics='mae')
    stockGRU.summary()
    #fitting the model and save the same
    history=stockGRU.fit(X_Train, y_train, epochs =200,batch_size=8,validation_data=(X_val, y_val))
    # #--------------------------
    # stockRNN = Sequential()
    # stockRNN.add(LSTM(units=100, return_sequences=True, input_shape=(X_Train.shape[1], X_Train.shape[2])))
    # stockRNN.add(LSTM(units=50, return_sequences=True))
    # stockRNN.add(LSTM(units=50))
    # stockRNN.add(Dense(units=1))
    # # compiling the model
    # stockRNN.compile(optimizer='adam', loss='mean_squared_error',metrics='mae')
    # stockRNN.summary()
    # # fitting the model and save the same
    # history1 = stockRNN.fit(X_Train, y_train, epochs=100, batch_size=8, validation_data=(X_val, y_val))
    # stockLSTM = Sequential()
    # stockLSTM.add(LSTM(units=100, return_sequences=True, input_shape=(X_Train.shape[1], X_Train.shape[2])))
    # stockLSTM.add(LSTM(units=50, return_sequences=True))
    # stockLSTM.add(LSTM(units=50))
    # stockLSTM.add(Dense(units=1))
    # # compiling the model
    # stockLSTM.compile(optimizer='adam', loss='mean_squared_error',metrics='mae')
    # stockLSTM.summary()
    # # fitting the model and save the same
    # history2 = stockLSTM.fit(X_Train, y_train, epochs=100, batch_size=8, validation_data=(X_val, y_val))
    #save the model
    stockGRU.save('models/20820331_RNN_model')
    # stockpred=Sequential()
    # stockpred.add(LSTM(units=100, return_sequences=True,input_shape=(X_Train.shape[1], X_Train.shape[2])))
    # stockpred.add(Dropout(0.2))
    # stockpred.add(LSTM(units=50, return_sequences=True))
    # stockpred.add(Dropout(0.2))
    # stockpred.add(LSTM(units=50))
    # stockpred.add(Dropout(0.2))
    # stockpred.add(Dense(units=1))
    # #compiling the model
    # stockpred.compile(optimizer='adam', loss='mean_squared_error', metrics='mae')
    # stockpred.summary()
    # # fitting the model and save the same
    # history1 = stockpred.fit(X_Train, y_train, epochs=120, batch_size=8, validation_data=(X_val, y_val))
    # stockRNN = Sequential()
    # stockRNN.add(SimpleRNN(units=100, return_sequences=True, input_shape=(X_Train.shape[1], X_Train.shape[2])))
    # stockRNN.add(Dropout(0.2))
    # stockRNN.add(SimpleRNN(units=50, return_sequences=True))
    # stockRNN.add(Dropout(0.2))
    # stockRNN.add(SimpleRNN(units=50))
    # stockRNN.add(Dropout(0.2))
    # stockRNN.add(Dense(units=1))
    # stockRNN.compile(optimizer='adam', loss='mean_squared_error', metrics='mae')
    # stockRNN.summary()
    # # fitting the model and save the same
    # history2=stockRNN.fit(X_Train, y_train, epochs=120, batch_size=8, validation_data=(X_val, y_val))
    # plotting the accuracy obtained by training and validation dataset
    # plt.plot(history.history['mae'])
    # plt.plot(history.history['val_mae'])
    # plt.plot(history1.history['mae'])
    # plt.plot(history1.history['val_mae'])
    # plt.plot(history2.history['mae'])
    # plt.plot(history2.history['val_mae'])
    # plt.title('model accuracy')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['GRU_train', 'GRU_val', 'RNN_train', 'RNN_val', 'LSTM_train', 'LSTM_val'], loc='upper left')
    # plt.show()
    # stockGRU = Sequential()
    # stockGRU.add(GRU(units=100, return_sequences=True, input_shape=(X_Train.shape[1], X_Train.shape[2])))
    # stockGRU.add(Dropout(0.2))
    # stockGRU.add(GRU(units=50, return_sequences=True))
    # stockGRU.add(Dropout(0.2))
    # stockGRU.add(GRU(units=50))
    # stockGRU.add(Dropout(0.2))
    # stockGRU.add(Dense(units=1))



