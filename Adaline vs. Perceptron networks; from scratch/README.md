# Perceptron vs. Adaline Classifier Comparison

This GitHub repository contains code and analysis for comparing the performance of Perceptron and Adaline classifiers on balanced and imbalanced datasets, as well as an investigation into the effect of learning rate on both classifiers.

## Dataset

### Balanced Dataset

In the balanced dataset, both classes have equal data points:
- Class 1: 100 data points
- Class 2: 100 data points

Both classifiers perform well on this dataset, achieving perfect classification. However, the Perceptron classifier demonstrates a larger margin as shown in the figure below.

![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/18d3be15-c634-4aad-b690-7c0cc196b40f)

### Effect of Learning Rate in Balanced Dataset

To study the effect of the learning rate on the classifiers in the balanced dataset, we kept the same random data and varied the learning rate. The figure below illustrates the results:

![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/73260725-8e88-4a3c-87a5-d70087254d92)

In the Perceptron network, changing the learning rate does not significantly impact the classification performance or the number of training epochs required. However, in the Adaline network, smaller learning rates (α) lead to better results. Notably, an α value of 0.1 achieves the desired outcome in just one epoch.

## imbalanced Dataset

In the imbalanced dataset, one class has significantly more data points than the other:
- Class 1: 100 data points
- Class 2: 10 data points

Given this class imbalance, we expect Adaline to perform less effectively than Perceptron.

As the initial weights of Adaline are randomly selected, the dividing line produced by the network may vary each time the program is run, resulting in different classification outcomes. Below are examples of linear classifiers produced by Adaline:
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/57097e0d-8239-42b7-8c2d-bd463ad9544a)

Run the classifiers again, and the result is:

![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/f44d86e8-e3fa-417f-ad91-13c744c34adc)

### Effect of Learning Rate in Imbalanced Dataset

In the imbalanced dataset, we examined the effect of changing the learning rate on both classifiers. The results are shown below:

![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/2ca10279-6c3b-4e67-bf11-bb3c0de0fd2c)

For the Perceptron network, changing the learning rate has little to no impact on the final weight formation, resulting in a stagnant classification boundary and consistent epoch counts.

Conversely, in the Adaline network, increasing the learning rate (α) leads to a reduction in the number of training epochs required for convergence.


