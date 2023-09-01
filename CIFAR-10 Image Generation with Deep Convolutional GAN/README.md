# CIFAR-10 Image Generation with InfoGAN

Welcome to our project on generating CIFAR-10 images using Information Maximizing Generative Adversarial Network (InfoGAN). InfoGAN is an extension of the GAN architecture that allows for controlled and meaningful image generation without the need for labeled data.

## About InfoGAN

InfoGAN is a GAN variant designed to generate images while providing control over specific attributes of those images. Unlike traditional GANs, InfoGAN introduces control variables in the latent space, enabling the network to manipulate features such as style, thickness, or type, even when class labels are unavailable. It achieves this by creating virtual labels within its latent space through clustering and conditioning the input noise signal.

### The Traditional GAN Challenge

Traditional GANs are highly effective at generating synthetic images but lack control over the generated content. Images are produced solely from random points in the latent space, making it challenging to influence specific image properties.

### The InfoGAN Solution

InfoGAN addresses this limitation by incorporating control variables alongside the latent space noise during image generation. By training the network with these control variables, the generator learns to manipulate certain properties of the generated images. This approach is especially valuable when desiring control over features like face shape, hair color, or hairstyle in image generation tasks.

### Learning Discrete Representations

InfoGAN is capable of learning discrete representations in an entirely unsupervised manner. Although the structure mapping within the generator may appear random, it enables control over features without requiring predefined labels.

## How InfoGAN Works

In InfoGAN, control variables and noise are provided as input to the generator. The model is trained using a cross-information loss function, encouraging it to learn interpretable and meaningful representations.

### Handling Different Variable Types

InfoGAN deals with categorical and continuous variables, each with different data distributions affecting mutual information loss calculations. The mutual loss is calculated and summed across all control variables based on variable types, ensuring a versatile approach to feature control.

## Generated Images by InfoGAN

In this section, we showcase the generated CIFAR-10 images at different training epochs to illustrate the progression of image quality and controlled feature manipulation by InfoGAN.

#### Epoch 10

<img src="https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/7dc885e3-0322-491e-89f7-2124645747eb" alt="Generated Images at Epoch 1" width="200"/>

#### Epoch 20
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/ed8046b5-a32c-47b7-aff7-00c64e2eb41a)


#### Epoch 50
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/8b5dc1b2-ee05-4967-b802-47730b42b69e)


#### Epoch 100

![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/f9243d30-12b3-4e55-b46d-90395c83152e)

### Epoch 200

![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/758ec81d-1032-4ee0-83d3-cc94cfa35fea)

