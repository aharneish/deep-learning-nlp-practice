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

2. Explore differnent activation functions. What are the pros and cons of different activation functions.

    * Activation functions play a crucial role in neural networks by introducing non-linearity and enabling the network to learn complex patterns from the input data.

     * The most commonly used activations functions are:

        a. Sigmoid:

                The sigmoid function is defined as $ f(x)= 1/{1+exp{^x}}  $.

                it squaches the input to a range between 0 and 1.

                * pros: 

                    * It provides a smooth gradient, making it suitable for gradient-based optimization algorithms.
                    
                    * It can be interpreted as a probability, making it useful for binary classification problems.
                
                * cons:

                    * Sigmoid activations suffer from the "vanishing gradient" problem. For very large or small inputs, the gradient becomes close to zero, leading to slow learning in deep networks.
                    * The output is not zero-centered, which can cause issues in network convergence.

        b. Tanh:
            
            * The hyperbolic tangent function, tanh(x), is similar to the sigmoid function but maps the input to a range between -1 and 1. 
            
            * Pros:

                * Like the sigmoid function, tanh provides a smooth gradient.
                
                * It is zero-centered, making it more convenient for optimization algorithms.
            
            * Cons:

                * Tanh activations also suffer from the vanishing gradient problem.
                
                * The output range is limited, which can result in slower convergence.
            
        c. ReLU (Rectified Linear Unit):
                
                * The rectified linear unit, ReLU(x) = max(0, x), is a popular activation function used in many deep neural networks. 

                * Pros:

                    * ReLU is computationally efficient to compute and allows for faster training compared to sigmoid and tanh.
                    
                    * It overcomes the vanishing gradient problem as it does not saturate for positive inputs.
                    
                    * ReLU activations provide sparsity in the network, as they zero out negative values.
                * Cons:

                    * The gradient for negative inputs is zero, which can cause "dying ReLU" issues where neurons become inactive and stop learning.
                    
                    * ReLU outputs are not bounded, which may result in exploding gradients during training.
        d. Leaky ReLU:
                
                *he Leaky ReLU is an extension of the ReLU function that addresses the dying ReLU problem. It introduces a small slope for negative inputs. Mathematically, it is defined as f(x) = max(ax, x), where a is a small constant. 
                
                * Pros:

                    *Leaky ReLU mitigates the dying ReLU problem by allowing small negative gradients for negative inputs.
                    
                    * It maintains the computational efficiency of ReLU.
                
                * Cons:

                    * Choosing the right slope parameter can be challenging, as it can impact the network's performance.
                    
                    * The output range is not bounded, which can result in exploding gradients.