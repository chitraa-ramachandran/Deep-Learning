from Model_MLP import MultiLayerPerceptron
import numpy as np
import pickle
#The train data and the train labels given are loaded using the genfromtxt functionality of numpy
X_TrainData = np.genfromtxt('train_data.csv', delimiter=',')
y_TrainData = np.genfromtxt('train_labels.csv', delimiter=',')
X_Train = X_TrainData[:20000]
Y_Train = y_TrainData[:20000]
X_Test = X_TrainData[20000:]
Y_Test = y_TrainData[20000:]
#The MLP object is called which will initialize the common variables
model = MultiLayerPerceptron(numOfEpochs=50, learningRate=0.01, InputSize=784)
#adding a hidden layer with a particular number of neuron
model.AddHiddenLayer(30)
#adding a output layer with a particular number of neuron
model.AddOutputLayer(4)
#initializing all the associated weights to zero
model.InitializeHiddenWeights()
model.InitializeOutputWeights()
#calling the fit function of the model
model.fit(X_Train, Y_Train)
#predicting the fitted model
pred = model.predict(X_Test)
#computing the accuracy of the model
accuracy = model.accuracy(pred, Y_Test)
print(accuracy)
#creation of the pickle file to store the weights and after storing the weights dump the model
pickle_file = "Final.pkl"
with open(pickle_file, 'wb') as file:
    pickle.dump(model, file)
