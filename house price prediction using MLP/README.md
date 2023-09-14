
## prediction of house price using MLP


In this section, we aim to predict house prices using the "House sales.csv" dataset, which is included with this file. Each row in this dataset contains information about a house, including the price of the house. We will begin by selecting the first 5000 rows from this dataset and then split the data, allocating 80% for training and 20% for testing.

### Step 1: Single Hidden Layer using MLP Network
- Utilize the information from this dataset to build a Multi-Layer Perceptron (MLP) neural network with a single hidden layer to predict house prices.
- Train the model using 80% of the data.
- Evaluate the model's performance on both the training and testing data.
- Report the error rate for both training and testing datasets.

### Step 2: Two Hidden Layers with Adjusted Neurons
- Enhance the neural network architecture by adding a second hidden layer.
- Experiment with different numbers of neurons in the two hidden layers.
- Train the model using the same 80% of the data.
- Evaluate the model's performance on both the training and testing data.
- Report the error rate for both training and testing datasets.


Please note that you should utilize the provided "House sales.csv" dataset for these tasks and provide error rate results for each step.


## Results
- normalize data
- loss: Mean squared error
- optimizer: adagrad
### MLP with one hidden layer
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/6875f406-7604-4055-9f1b-1bcc85172047)


### MLP with two hidden layer

![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/656f864e-68cd-4695-bb18-75420614459c)
