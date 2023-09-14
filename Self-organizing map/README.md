## Project Description
Load the Fashion-MNIST dataset from keras according to the following command.
`from keras.datasets import fashion_mnist`

Cluster the first 1000 data points of the dataset using a SOM network consisting of 841 neurons according to the mentioned sections. (The Fashion-MNIST dataset consists of 28x28 photos of clothing classified into 10 classes.)

1) Set the proximity radius of each neuron to zero.
2) Place the neurons in a linear form with a proximity radius of R=1.
3) Place the neurons on the nodes of a 29 x 29 grid with a square adjacency form with a radius of R=1.
