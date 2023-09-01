# Generating Hand Written Digits using Deep Learning

Welcome to our project focused on generating hand-written digits using Variational Autoencoder networks, often referred to as VAEs. VAEs are powerful generative models used for data generation, consisting of two main components: the encoder and the decoder.

Note: The readme is translated from my Persian report into English by Google Translate and revised by ChatGPT.
## Project Description

The primary distinction between Variational Autoencoder networks and regular autoencoders lies in the encoder section. Unlike traditional autoencoders, VAEs generate two vectors in the encoder: one containing means and another containing standard deviations. In the decoder section, after sampling these average and standard deviation vectors, the output is produced. Unlike autoencoders, VAEs do not aim to reproduce input images exactly. Instead, they generate new images, performing statistical manipulations in such a way that the generated patterns are different from the original patterns in the encoding part, yet belong to the same category.

VAEs are formidable contenders in generative models and find applications in image-to-image and text-to-image translation, music generation, fake image production, text generation, and more. They are particularly useful when control over the generation process is desired. In some cases, VAEs can even outperform GANs in terms of control and usability.

The key to VAE's power lies in the central part of the autoencoder, where the encoded space has the most variance. By allowing objects with attributes and classes to apply different forms and features separately in this central coded space (e.g., representing features of digits in MNIST), we can create combinations between these features, resulting in versatile pattern generation.

## Comparing GANs and VAEs

The process of data generation and the advantages and disadvantages of GANs (Generative Adversarial Networks) and VAEs compared to each other differ significantly:

### GANs
- GANs are based on the conflict between the Discriminator and Generator.
- The Discriminator learns to distinguish between real and fake data, acting as a binary classifier.
- The Generator learns to generate samples from independent and identically distributed (iid) noise.
- As training progresses, the Generator creates more convincing fake samples, making its productions closer to reality.

### VAEs
- VAEs generate new data by minimizing a loss function.
- The Encoder maps real samples to a latent space with mean and variance for each dimension.
- Samples are drawn from these density functions and fed into the Decoder to generate new samples.
- VAEs involve a semi-supervised learning process and aim to minimize a loss function to create new data.

### Comparison
- GANs typically require more training time than VAEs but are often more stable.
- In cases with a low variety of classes in the dataset (e.g., MNIST), both methods can produce good images.
- VAEs offer statistical interpretability, allowing the modeling of probability density functions, unlike GANs.
- VAEs excel when adding features to images is desired.

## Implementation of VAE Model using MNIST Dataset

In this section, we implement a VAE model using the MNIST dataset. We provide the implementation for different IPACs (Interpolation and Extrapolation of Probabilities and Active Contexts). Detailed explanations and code are available in the attached notebook file.

For each model used to generate data, we showcase results for 5 different IPACs. The network structure is visualized in the figures below:

#### Encoder Structure
  
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/020d7229-d89f-4d60-99ab-e5abfc4dcc6c)

#### Decoder Structure

  
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/f9f56d8d-8cef-498c-a5c9-11ee3902f6d8)


### Train and validation

The following figures depict the generated images and loss values in epochs 20, 50, 80, 100, and 150:

#### Epoch 20
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/2915bcff-26f4-418b-b143-cd60a655b78e)
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/542ce405-d622-46b3-bf84-7c9140d87ee0)

#### Epoch 50


![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/de49312f-b1f4-479c-b0b0-9d1283a4f21d)
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/10620aca-cfb3-4a44-8ed7-511321b76bed)

#### Epoch 80
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/b493c96f-ac3e-4724-b2e6-d10cf370717a)
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/3e9731d0-9926-4ced-a749-0d931787d4c0)


#### Epoch 100
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/0c75e55d-6ec2-42ad-8ee8-0128fe1ab7f1)
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/823e8c78-fa8b-4b6f-812e-1d1dba3786f2)


#### Epoch 150
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/75f48bec-b038-4118-aa41-3749a4978189)
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/b3321f4b-cecf-4412-a64d-a479e308e60e)

## Clustering Validation data in 2D space
With the help of Encoder, we move the Validation data images from the 784 dimensional input space to the 2D space and cluster them. The results are as follows

#### Epoch 20
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/430b83cd-a78f-4b4a-a80a-a26fb1202477)

#### Epoch 80 
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/fb8c6cb6-972a-4ba2-ad32-f464bc0e29c3)


#### Epoch 150
![image](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/4d9db7ac-8cd3-454b-a5c4-7560607ea94b=300*300)

