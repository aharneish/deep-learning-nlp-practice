1. understand RNNs (including the mathematical notations)

   *  A recurret neural network (RNN) is a type of artificial neural network which uses sequential data or time series data.

   * Commonly used for solving problems such as language translation, natural language processing (nlp), speech recognition and image captioning.

   * NNs are a type of neural network that can be used to model sequence data. RNNs, which are formed from feedforward networks, are similar to human brains in their behaviour.

   * RNN uses a hidden layer to overcome the problem of remembering the previous input.

   * The most important component of RNN is the Hidden state, which remembers specific information about a sequence.

   * RNNs have a Memory that stores all information about the calculations. It employs the same settings for each input since it produces the same outcome by performing the same task on all inputs or hidden layers.

   * RNNs are type of neural network that has hidden states and allows past outputs to be used as inputs. 

![rnn2](https://github.com/aharneish/deep-learning-nlp-practice/assets/99192645/07f1d71b-fdcf-4453-801d-af64d6d50b7e)

   * RNN architecture depends on the type of problem it is being used to solve.

   * There can be:

         1. one to one

         2. one to many

         3. many to one

         4. many to many

   * The information in the recurrent neural networks cycles through a loop to the meiddle hidden layer.

   * The input layer **x** recieves and processes the neural network's input before passing it onto the middle layer.

   * Multiple hidden layers can be found in the middle layer __h__, each with its own activation functions, weights, and biases.

   * The basic issues with RNNs are:

         1. exploding gradients

         2. vanishing gradients.

2. Understand CBOW and Skip-Gram.

   * CBOW (Continious Bag Of Words) and Skip-Gram are popular alogrithms in Natural Language Processing for generating word embeddings.

   * They are part of the Word2Vec framework.

   * CBOW predicts a target word based on its surrounding context word.

      * Input: Given a target word CBOW takes a fixed number of context words as input

      * Embedding layer: Each context word is converted into a dense vector representation.

      * Aggregation: The word embeddings of the context words are averaged or summed to create a single context vector.

      * Output: The context vector is used as input to a softmax layer which predicts the target word.

   * CBOW is computationally efficient and tends to work well for frequent words and when the dataset is large. However, it may not capture rare word representations effectively.

   * Skip-Gram, in contrast to CBOW, predicts the context words given a target word. 

      * Input: Given a target word, Skip-Gram takes the target word's embedding as input.

      * Output: The target word embedding is fed into a softmax layer, which predicts the probability distribution of the context words.

   * Skip-Gram is more effective for capturing semantic relationships and performing well with rare words, but it can be computationally expensive compared to CBOW.