1. Explore AlexNet, VGGnet architectures
    
    AlexNet:
        
        AlexNet is a pioneering convolutional neural network (CNN) architecture that gained significant attention when it won the ImageNet Large Scale Visual Recognition Challenge in 2012. It consists of eight layers, including five convolutional layers and three fully connected layers. Here's a breakdown of its key components:
            
            * Convolutional Layers: The convolutional layers in AlexNet perform the main feature extraction. They use small filters with different depths and sizes to convolve across the input image and generate feature maps. The depths of the convolutional layers increase as the network goes deeper, allowing for the extraction of increasingly complex features.
            
            * ReLU Activation: AlexNet popularized the use of the Rectified Linear Unit (ReLU) activation function. ReLU introduces non-linearity to the network and helps capture complex patterns in the data.
            
            * Max Pooling: Max pooling layers follow some of the convolutional layers. They downsample the feature maps, reducing their spatial dimensions while retaining the most important features.
            
            * Local Response Normalization: This technique normalizes the responses of neurons within a local neighborhood. It enhances the network's ability to generalize and reduces the chances of overfitting.
            
            * Fully Connected Layers: The final layers of AlexNet are fully connected layers that aggregate the extracted features and produce the classification output.
    
    VGGNet:
       
        VGGNet, proposed in 2014, is known for its simplicity and deep architecture. It consists of 16 or 19 layers and follows a uniform structure throughout the network. Here's a closer look at its key aspects:
          
            * Convolutional Layers: VGGNet employs a series of convolutional layers with small 3x3 filters and a stride of 1. These layers are stacked on top of each other, resulting in a deep network. The small filter size allows for more detailed feature extraction.

            * Max Pooling: Similar to AlexNet, VGGNet uses max pooling layers to reduce spatial dimensions and retain important features.

            * Deep Structure: VGGNet is deeper than AlexNet, with up to 19 layers. The increased depth allows the network to learn more complex representations but also increases computational complexity.

            * Transfer Learning: VGGNet has been widely adopted for transfer learning due to its pre-trained models. By using pre-trained weights on large-scale datasets, VGGNet models can be fine-tuned for specific tasks with limited data.

2. Explore differnent activation functions. What are the pros and cons of different activation functions