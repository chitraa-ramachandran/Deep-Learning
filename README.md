# Deep-Learning
### MLP WITH BACKPROPAGATION
A multilayer perceptron (MLP) is a class of feed forward neural network. An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function.MLP utilizes a supervised learning technique called backpropagation for training. Since MLPs are fully connected, each node in one layer connects with a certain weight to every node in the following layer. 

Learning occurs in the perceptron by changing connection weights after each piece of data is processed, based on the amount of error in the output compared to the expected result. The principle of the backpropagation approach is to model a given function by modifying internal weightings of input signals to produce an expected output signal. The system is trained in such a way that the error between the system’s output and a known expected output is presented to the system and used to modify its internal state.

Backpropagation can be used for both classification and regression problems

#### IMPLEMENTING BACKPROPAGATION ALGORITHM FROM SCRATCH 
<br /> 

We are considering a training dataset with 24754 samples with 784 features which can be classified into four different classes (0,1,2,3). The labels are hot encoded. 
For the purpose of creating a back-propagation model of a multi-layer perceptron we are dividing our training dataset into train data and validation data in the ratio of 8:2. <br /> 
Reason for creating a validation dataset: 
The evaluation of a model’s ability on a training dataset is biased. Hence we need a unbiased estimate of our model’s ability.So we hold back a particular portion of the training dataset and use them to test the model which in turn will help us to tune the hyper parameters of our model. 

#### Creation of Model instance: <br /> 
The MLP constructed has one input layer, one hidden layer and one output layer.The inputs given to our model are : <br /> 
•	•	●  Input Size <br /> 
•	•	●  No of Epochs <br /> 
•	•	●  Learning Rate <br /> 
 <br /> 
Learning Rate(Step size) is the amount by which the weights are updated during each iteration of training.Learning rate is usually considered to be a very small value in terms of 0.1,0.01,0.001 etc.We need to choose an optimal value for the learning rate such that the training doesn't become extremely slow.Based on the experiments I tried am assuming the learning rate to be 0.01 as the loss value gets stabilized and the training also happens reliably faster when compared to a learning rate of 0.001. 
<br /> 
 <br /> 
Epochs is the total number of passes the training data is considered to fit the model.Higher the number of epochs considered higher will be the accuracy until and unless the model is not getting overfitted. <br /> 
Initializing the weights: <br /> 
We have two sets of weight matrices associated with our model of MLP.We initialize the weights that are between the Input and the hidden layer as ‘0’s and similarly the weights between hidden and the output layer to be ‘0’s 
The weights associated between these layers depend on the input size to the layer and the number of neurons in the next connected layer.
<br /> 
#### FITTING THE MODEL AND DOING PREDICTION: <br /> 
The model is fitted on the training dataset where the feed forward pass and the back propagation happens through all the layer for the mentioned number of epochs.The fitted model will now predict the result of X in the validation dataset.The obtained result is compared with the original Y value in order to calculate the accuracy and the average error which is calculated using the cross entropy loss function.Using the obtained accuracy and error value we need to tune the hyperparameters of the model in such a way that the accuracy is improvised and the error value is minimized. 
We are using the softmax loss function or the cross entropy loss function which will measure the performance of the classification model.It is considered to be the natural loss function to be used if we use a sigmoid or a softmax nonlinearity in the output layer of the network. <br /> 
#### ACTIVATION FUNCTIONS USED : <br /> 
Normally the output layer utilizes softmax activation function and the hidden layer utilizes the sigmoid/tanh/ReLu activation function and upon trial and error i have assumed Sigmoid activation function in the hidden layer as it gave a better accuracy than the others. <br /> 
The softmax function uses exponential function which increases the probability of the maximum value of the previous layer and makes sure that for a given classification problem the summation of all the output will be equal to 1.This is the main significance of using a softmax activation function in the output layer. 
I have decided on the number of epochs and the learning rate based on accuracy and loss function.


### RBNF NEURAL NETWORK
Radial Basis Function Neural Network is one of the fast, effective and intuitive Machine Learning algorithms.3-layered network that uses radial activation function and can be used to solve both classification and regression problems.We have considered a custom mapping function to create 441 training data.It uses a hybrid approach for training the model.
<br />
The network structure uses nonlinear transformations in its hidden layer (typical transfer functions for hidden functions are Gaussian curves). However, it uses linear transformations between the hidden and output layers. The rationale behind this is that input spaces, cast nonlinearly into high-dimensional domains, are more likely to be linearly separable than those cast into low-dimensional ones. 
<br />
The nonlinear transformations at the hidden layer level have the main characteristics of being symmetrical. They also attain their maximum at the function center, and generate positive values that are rapidly decreasing with the distance from the center.To determine the centers for the RBF networks, typically unsupervised training procedures of clustering are used and I have used K-Means clustering technique.Once the centers and the widths of radial basis functions are obtained, the next stage of the training begins where we use gaussian kernel function and use pseudo inverse method to find the weight matrix of the network.
<br />

### SELF ORGANIZING MAPS
Kohenen's SOM belongs to the class of unsupervised learning network.The nodes of the KSOM can recognize groups of similar input vectors.This generates a topographic mapping of the input vectors to the output layer, which depends primarily on the pattern of the input vectors and results in dimensionality reduction of the input space.So,we consider a input vector that is connected to the neurons in the topographical map by weights and through the process of competitive learning we find the winning neuron and after which we update the weights of the neighbourhood of the winning neuron.The network is trained on a certain number of epochs and as the number of epochs through which the model is trained increases the neighbourhood also decreases and the self organizing network convergers in finite steps.The goal is that if you present a similar input vector to the SOM it should be able to recognize the stored input pattern presented to it during the training and match the present input vector to the desired target neuron.
<br />
<br/> <br />
The analysis and report with full code is present in the uploaded ipynb files.


