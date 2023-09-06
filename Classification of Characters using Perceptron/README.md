
# OCR Perception Network

This repository contains code and data for training and testing an Optical Character Recognition (OCR) Perception Network. The network is designed to recognize a set of 21 letters based on numerical representations provided in the "OCR_train.txt" file. Each letter is represented by a 9x7 matrix and a label.

## Dataset

The dataset used for training and testing the OCR Perception Network is defined in the "OCR_train.txt" file. This file contains 21 lines, each representing one of the 21 letters. Each line consists of 71 numbers, where:

- The first 63 numbers form a 9x7 matrix representing the letter's shape.
- The 65th to 71st numbers represent the label of the letter.
- The 64th number in all lines is equal to one, serving as a separator between the main part and the letter labels.

In the figure below, the first line of the file (which belongs to the letter A) is given:
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/d2158004-f509-4360-ad8e-db87e4db53e5)


## Training and Testing

The OCR Perception Network is trained and tested in two scenarios:

### Scenario 1: 21 Train Data Points and 71 Test Data Points

In this scenario, the network is trained with 21 data points and tested with 71 data points. The initial weights and biases are set to zero. Different values of learning rate (α) and threshold (θ) were experimented with, resulting in the following outcomes:

- Training with α=0.1 and θ=100: 73 epochs, Error = 0.2571
- Training with α=0.4 and θ=100: 19 epochs, Error = 0.2714
- Training with α=0.7 and θ=100: 13 epochs, Error = 0.3143
- Training with α=1 and θ=100: 12 epochs, Error = 0.3143
- Training with α=0.1 and θ=150: 107 epochs, Error = 0.2571
- Training with α=0.4 and θ=150: 27 epochs, Error = 0.2714
- Training with α=0.7 and θ=150: 21 epochs, Error = 0.3
- Training with α=1 and θ=150: 13 epochs, Error = 0.3143
- Training with α=0.1 and θ=200: 143 epochs, Error = 0.2571
- Training with α=0.4 and θ=200: 40 epochs, Error = 0.2571
- Training with α=0.7 and θ=200: 24 epochs, Error = 0.3
- Training with α=1 and θ=200: 19 epochs, Error = 0.3143

In this scenario, the best performance achieved is a 25.7% error rate.

### Scenario 2: 71 Train Data Points and 21 Test Data Points

ُُThis scenario involves training the network with 71 test data points and testing it with 21 train data points. The results are as follows:

- Training with α=1 and θ=0.02: 12 epochs, Error = 0.0

