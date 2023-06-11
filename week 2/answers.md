1. Difference between a neural network with all the neurons in a single layer vs neurons stacked in multiple layers?

a. In a neural network with multiple neurons in a single layer or the dense neural network, the network is able to learn the intricate representations of the input data taken from the previous layer.
b. The model complexity is limited by the number of neurons available in the single layer. It is simple to design; it is computationally effective and it is easily interpretable.
c. In a neural network with distributed neurons which is also called a deep neural network; the layers are arranged in a way to process the data sequentially; it is able to learn abstract and the complex representations of the data.
d. The model is useful in hierarchical feature extraction; we can increase the model capacity by stacking multiple layers over the previous layers giving improved performance; it helps in generalization tasks.

2. Difference between Generative and Discriminative neural networks?

a. A generative model aims to understand and model the underlying distribution of the input data in order to generate new samples that resemble the original distribution.
b. These models learn from the joint probability distribution of the input data and labels. Given this learned distribution it can generate new samples that exhibit similar characteristics to the training data.
c. It can be used for data generation, unconditional and conditional generation of data.
d. Discriminative models focus on learning the decision boundary between different classes or categories of the input data.
e. The model learns the conditional probability of the labels given the input data.
f. These models aim to directly lean the conditional between classes rather than generating the new samples.
g. It can be used for making class-specific decision making, object detection and segmentation.
h. These models have generally simple architectures and are computationally efficient.

3. What is cross entropy loss?

a. It is a commonly used loss function in machine learning, particularly in classification tasks.
b. It quantifies the dissimilarity between the predicted probability distribution and the true label distribution of the training data.
c. It measures the average number if bits needed to encode the true labels based on the predicted probabilities.
d. By minimizing the cross-entropy loss, the model is encouraged to assign higher probabilities to correct classes and lower probabilities to the incorrect classes.
 
4. What is KL divergence?

a. KL divergence short for Kullaback-Leibler divergence, is a measure of how one probability distribution differs from another.
b. It is often used in information theory and machine learning to quantify the dissimilarity between two probability distribution.
c. KL divergence is not a symmetric measure, meaning the KL divergence from distribution A to distribution B is not the same as the KL divergence from B to A.
d. KL divergence is always non-negative, it is not symmetric, it not a true distance metric. 

5. What is the relation between KL divergence and entropy?

a. While KL divergence quantifies the difference between two distributions, entropy measures the uncertainty or average amount of information in a single distribution.
b. KL divergence can be decomposed into the difference between the entropy of P and the cross entropy of P and Q.
c. Can be represented mathematically as:
KL(P||Q)=H(P)-H(P,Q)

