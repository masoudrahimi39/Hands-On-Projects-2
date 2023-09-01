# CIFAR-10 Image Generation with Conditional GAN (cGAN)

In this project, we explore the fascinating world of Conditional Generative Adversarial Networks (cGANs) and their application to image generation using the CIFAR-10 dataset. cGANs enhance traditional GANs by leveraging class labels to generate data with precision and control. This approach leads to improved network performance, resulting in more stable training, faster convergence, and the creation of high-quality images.

## About Conditional GAN

Conditional Generative Adversarial Networks (cGANs) share the fundamental architecture of GANs, but with a significant difference. In cGANs, we utilize class labels to guide data generation. This means we can specify the label for the generated data, resulting in improved network performance. Enhancements include more stable training, faster convergence, and higher-quality image generation.

### cGAN Architecture

The cGAN architecture consists of two key components: the **Generator** and the **Discriminator**.



#### Generator

The Generator begins with a point in latent space as its input. Utilizing convolutional layers, it transforms this input into multiple low-resolution copies, specifically, generating 128 (128 was selected by me) copies of 7x7 images (7x7 resolution for the low-resolution output image was selected by me). To incorporate class labels, represented as integers, we pass them through an embedding layer. This process converts each label into a vector. Subsequently, we employ a fully connected layer before reshaping it into a feature map with dimensions, for instance, 7x7. At this stage, we obtain a (7,7,1) feature map, which serves as our low-resolution output image.

The next step involves concatenating the information derived from the class label with the Generator's input, resulting in a (7,7,129) feature map. Finally, we utilized this information within the Generator, which employs convolutional layers to produce a 28x28 image as the final output.

#### Discriminator

The Discriminator begins by taking a 28x28 image as input and promptly provides an output of either 0 or 1. Its primary task is to determine whether the image is real or fake. To incorporate class labels, represented as integers, we pass them through an embedding layer, transforming each label into a vector. Subsequently, a fully connected layer is employed. For instance, if the images are 28x28 pixels in size, the dense layer comprises 784 (28x28) neurons. Following this, we reshape the dense layer to match the 28x28 image dimensions.

This reshaping process effectively adds the class label, which is now the same size as the image, as an additional channel to the feature map. Subsequently, by utilizing convolutional and fully connected layers, the Discriminator produces an output of 0 or 1.


## Harnessing the Power of cGANs

In this project, we harness the capabilities of Conditional Generative Adversarial Networks (cGANs) to generate high-quality images using the CIFAR-10 dataset. Our exploration delves into the training process, evaluation metrics, and practical utilization of the trained model for image generation.

#### Complete Model

![c](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/4534ba5d-c028-40cb-9c3d-5cf256dcc513)


#### Generator Model
![a](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/cd30fa98-c22d-4739-9b0f-42e772f01b69)

#### Discriminator Model
![b](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/19a15ea2-fb4a-4830-8108-463366f470c8)

#### Original Images
![orginal](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/8e7fba10-15ab-432e-9c1d-72b2e1a7493e)


#### Evolution of Generated Images in Different Epochs
##### Epoch 1
![1](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/ed11bbb3-c961-4f2f-a132-6cd4a46f4304)

##### Epoch 11
![11](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/91c83945-f775-45df-860a-2a7180cf5633)

##### Epoch 21
![21](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/a4d2efa1-6de9-4363-8c7a-b6139c06198a)

##### Epoch 31
![31](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/059a718b-dd99-42f1-b2fe-2260ed05e7b0)

##### Epoch 41
![41](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/e8821ad5-f88f-4be0-bd46-2f53abb11aaf)

##### Epoch 51
![51](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/3aa63a6d-6614-4cd3-b223-4e91c8ffa9ff)

##### Epoch 61
![61](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/bf7ffbbc-3989-442e-ad28-fad8fe8512a4)

##### Epoch 71
![71](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/aaec3f5b-85a7-4ce0-848e-b91f6bcfef5a)

##### Epoch 81
![81](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/d1ae5807-54d8-4b12-97aa-c7430964ff1f)

##### Epoch 91
![91](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/602124f3-43f7-410c-a2c4-726a6b0c13e6)


#### Training Loss
![loss](https://github.com/masoudrahimi39/Machine-Learning-Hands-On-Projects/assets/65596290/2e5c30bf-8457-4ede-bbc9-656899dec76c)


