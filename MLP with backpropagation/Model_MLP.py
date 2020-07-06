import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
# creating class MultiLayerPerceptron with the initial data members and functions
class MultiLayerPerceptron():
    #init initializes all the neccessary data members that will be used by different methods of the class MultiLayer Perceptron
    def __init__(self, numOfEpochs=1, learningRate=0.01, InputSize=1):
        self.numOfEpochs = numOfEpochs
        self.learningRate = learningRate
        self.InputSize = InputSize
        self.HiddenLayerNeurons = 0
        self.OutputLayerNeurons = 0
        self.HiddenWeights = []
        self.OutputWeights = []
        self.HiddenLayer = []
        self.OutputLayer = []
        self.OutputLayerInput = []
        self.OutputLayerActivate = []
        self.HiddenLayerActivate = []
        self.InputBias = []
        self.iter = []
        self.acc_list = []
        self.loss_list = []
    #The below function adds a hidden layer to the network with a particular number of neurons
    def AddHiddenLayer(self, Neurons):
        self.HiddenLayerNeurons = Neurons
    # The below function adds a output layer to the network with a particular number of neurons
    def AddOutputLayer(self, Neurons):
        self.OutputLayerNeurons = Neurons
    #initialize the weights of the links to the hidden layer as 0 to start with the first feed forward pass
    def InitializeHiddenWeights(self):
        self.HiddenWeights = np.full((self.InputSize+1,self.HiddenLayerNeurons), 0)
    # initialize the weights of the links to the output layer as 0 to start with the first feed forward pass
    def InitializeOutputWeights(self):
        self.OutputWeights = np.full((self.HiddenLayerNeurons+1,self.OutputLayerNeurons) , 0)
    #Sigmoid activation function returning a sigmoidal value "used as the activation function in the hidden layer"
    def sigmoid(self, val):
        sigmoid_out = 1 / (1 + np.exp(-val))
        return sigmoid_out
    #Softmax activation function returning an exponential value "used as the activation function in the output layer"
    def softmax(self, val):
        exponential = np.exp(val)
        softmax_out = exponential/np.sum(exponential, axis=0)
        return softmax_out
    #Feed forward pass of the constructed MLP
    def FeedForward(self,X):
        self.InputBias = np.append(X,[-1])
        # Ouput that has the sum of the weighted inputs with the bias
        self.HiddenLayer = np.matmul(self.InputBias, self.HiddenWeights)
        # Hidden layer output is passed through the the sigmoid function to get the final hidden value output
        self.HiddenLayerActivate = self.sigmoid(self.HiddenLayer)
        self.OutputLayerInput = np.append(self.HiddenLayerActivate,[-1])
        #Ouput that has the sum of the weighted inputs with the bias
        self.OutputLayer = np.matmul(self.OutputLayerInput, self.OutputWeights)
        #the output is passed through the softmax activation function to get the final output
        self.OutputLayerActivate = self.softmax(self.OutputLayer)
        return self.OutputLayerActivate
    #Backpropagation algorithm that does error correction and weight updation at a particular learning rate
    def BackPropogation(self,y):
        #the delta value to be backpropagated to the hidden layer
        self.deltaOutput = self.OutputLayerActivate - y
        self.deltaOutput = np.asarray(self.deltaOutput)
        self.deltaOutput = self.deltaOutput.reshape(-1, len(self.deltaOutput))
        self.OutputLayerInput = self.OutputLayerInput.reshape(-1, len(self.OutputLayerInput))
        self.deltaWeightOutput = np.matmul(np.transpose(self.deltaOutput), self.OutputLayerInput)
        # the delta value to be backpropagated to the input layer
        self.deltaHidden = np.matmul(self.deltaOutput, np.transpose(self.OutputWeights[:len(self.OutputWeights)-1]))
        self.deltaHidden = self.deltaHidden * self.HiddenLayerActivate * (1 - self.HiddenLayerActivate)
        self.InputBias = np.asarray(self.InputBias)
        self.InputBias = self.InputBias.reshape(-1, len(self.InputBias))
        self.deltaWeightHidden = np.matmul(np.transpose(self.deltaHidden), self.InputBias)
        #updation of the associated weights
        self.HiddenWeights = self.HiddenWeights - self.learningRate*np.transpose(self.deltaWeightHidden)
        self.OutputWeights = self.OutputWeights - self.learningRate*np.transpose(self.deltaWeightOutput)
    #Considering Cross entropy loss function as the MLP's loss function
    def Cross_Entropy_Loss(self,y_obtained, y_actual):
        sum_of_loss = np.sum(np.multiply(y_actual, np.log(y_obtained)))
        total = len(y_obtained)
        loss = -(1/total) * sum_of_loss
        return loss
    #Fitting the model on the training dataset and each training dataset is run for a particular number of epoch
    def fit(self,X,y):
        error = 0
        self.validation_accuracy = 0
        for i in range(self.numOfEpochs):
            randomized_val = np.arange(len(X))
            np.random.shuffle(randomized_val)
            X=X[randomized_val]
            y=y[randomized_val]
            X_TrainData=X[:16000]
            y_TrainData=y[:16000]
            X_TestData=X[16000:]
            y_TestData=y[16000:]
            self.y_pred=[]
            sum_error = 0
            for j in range(len(X_TrainData)):
                obtained = self.FeedForward(X_TrainData[j])
                self.BackPropogation(y_TrainData[j])
                target = y_TrainData[j]
                error = self.Cross_Entropy_Loss(obtained, target)
                sum_error += error#the error for each datapoint is cumulated to find the average error at the next stage
            Average_error = sum_error/len(X_TrainData)
            self.y_pred=self.predict(X_TestData)
            self.validation_accuracy=self.accuracy(self.y_pred,y_TestData)
            #the below list are appended for plotting and understanding the hyperparameter tuning by trail and error method
            self.iter.append(i)
            self.acc_list.append(self.validation_accuracy)
            self.loss_list.append(Average_error)
            print("Epoch ",i," Loss ",Average_error," Accuracy ",self.validation_accuracy)
        #The plots are displayed
        plt.plot(self.iter, self.loss_list)
        plt.show()
        plt.plot(self.iter, self.acc_list)
        plt.show()
    #Predicting the fitted model on validation or test dataset
    def predict(self,X):
        self.Prediction_Array = []
        Value = 0
        for i in range(len(X)):
            self.Output = self.FeedForward(X[i])
            Value = np.argmax(self.Output)
            self.predicts = np.zeros(len(self.Output))
            self.predicts[Value] = 1
            self.Prediction_Array.append(self.predicts)
        self.Prediction_Array = np.asarray(self.Prediction_Array)
        return self.Prediction_Array
    #Computing the accuracy by comparing the obtained y value with the original target y value
    def accuracy(self, y_obtained, y_original):
        sum = 0

        for i in range(len(y_obtained)):
            y_pred = np.argmax(y_obtained[i])
            y_actual = np.argmax(y_original[i])
            if (y_pred == y_actual):
                sum += 1
        accuracy = sum/len(y_obtained)
        return accuracy