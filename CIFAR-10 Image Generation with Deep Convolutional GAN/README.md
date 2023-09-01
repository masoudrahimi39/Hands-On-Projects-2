
# CIFAR-10 Image Generation with Deep Convolutional GAN

In this project, we delve into the realm of Generative Adversarial Networks (GANs). GANs operate by pitting two distinct neural networks against each other: the Generator and the Discriminator. These networks engage in a competitive and cooperative dance, each striving to excel without the need for explicit supervision.

At the heart of GANs lies a simple yet challenging concept: one network learns to create authentic-looking images (Generator), while the other learns to distinguish between genuine and synthetic images (Discriminator). One of the most well-known GAN architectures is the DCGAN model, which we aim to implement in this project.

## DCGAN Architecture Design Mechanism

GANs represent a captivating idea in computer science, introduced by Goodfellow, that addresses unsupervised problems through two deep networks: Generator and Discriminator. These networks engage in a constant interplay, learning simultaneously but in opposite directions. One network's goal is to generate authentic images, while the other's is to spot the fakes.

### Key Elements of DCGAN:

1. **Strided Convolution and Fractional Strided Convolution:** In DCGAN, max-pooling layers in traditional CNNs are replaced with strided convolution in the Discriminator and fractional strides convolution in the Generator. This allows the network to learn both downsampling and upsampling, serving different purposes in each section.

2. **Batch Normalization:** Batch normalization is applied after each convolutional layer in both the Generator and Discriminator. This normalization helps stabilize the learning process and ensures that the network doesn't converge to a single point of failure, particularly crucial in the Generator network.

3. **Activation Functions:** The Generator network employs Rectified Linear Unit (ReLU) for all layers except the final one, which utilizes the Tanh function for faster convergence. The Discriminator network employs Leaky ReLU activation functions for all layers, enhancing its performance with high-resolution images.

4. **Fully Connected Layer Removal:** DCGANs eliminate the fully connected layer at the beginning of the Generator and at the end of the Discriminator. Instead, noise is injected into the Generator as a 100-dimensional array, which is then reshaped and fed into the subsequent convolutional layers.

5. **Noise Injection:** To mitigate issues like vanishing gradients and unstable generator gradient updates, noise is added to the input of the Discriminator. This introduces a smoothing effect to the probability distribution and maintains non-zero Jensen-Shannon Divergence (JSD) values, ensuring stable learning.

6. **Optimizer - Adam:** Adam, a hybrid of Momentum and RMSProp, is employed as the optimizer. It combines the advantages of both methods, ensuring faster convergence while preventing oscillations around local minima.

**Strided Convolution and Fractional Strides Convolution**

Strided convolution and fractional strides convolution are pivotal components of DCGAN architecture. Strided convolution replaces max-pooling layers, enabling down-sampling in the Discriminator and up-sampling in the Generator.

These architectural innovations are essential to the success of DCGANs in generating high-quality images. They allow GANs to operate more effectively on image data, overcoming many challenges posed by traditional CNNs.

## Implementation of DCGAN model using CIFAR10 dataset

In this section, we describe the implementation of the DCGAN model using the CIFAR10 dataset. We utilize the CIFAR10 dataset to generate images, showcasing the capabilities of our DCGAN architecture.

#### Epoch 20 
<img src="https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/f4fca2d4-b3aa-4638-ad20-606097085044" alt="Epoch 20" width="300" height="300">

#### Epoch 40
<img src="https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/74d27548-74db-4a06-ade6-0dbcbe898db6" alt="Epoch 40" width="300" height="300">

#### Epoch 60
<img src="https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/56b6ed77-988b-493c-8f5d-9b52ccd7c57b" alt="Epoch 60" width="300" height="300">


#### Epoch 80
<img src="https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/a9bbb5ee-fe0a-49f3-8105-bea96a5b55f7" alt="Epoch 80" width="300" height="300">

#### Epoch 100
<img src="https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/1093e28c-1335-4977-b6b0-42988d18da63" alt="Epoch 100" width="300" height="300">
<img src="https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/7f2004bb-86fc-41f0-b802-1555e0ce9c25" alt="Epoch 100" width="300" height="300">
<img src="https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/d93f90f5-8892-4f7e-96da-7c6e5f8112a1" alt="Epoch 100" width="300" height="300">



