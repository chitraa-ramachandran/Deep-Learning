import numpy as np
import pickle
from Model_MLP import MultiLayerPerceptron
STUDENT_NAME = 'CHITRAA RAMACHANDRAN'
STUDENT_ID = '20820331'

def test_mlp(data_file):
	#Reading the test data file
	X_Test = np.genfromtxt(data_file, delimiter=',')
	#Loading the generated pickle file
	pickle_file = "Final.pkl"
	with open(pickle_file, 'rb') as file:
		pickle_instance = pickle.load(file)
	#calling the predict instance with the obtained weight
	prediction = pickle_instance.predict(X_Test)
	return prediction
# prediction=test_mlp('./train_data.csv')
# print(prediction)
'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./train_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''
