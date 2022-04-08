This project implements a 2-step classification network in order to find a relationship between the text and numerical reviews. The first step is to use a pretrained model to transform the verbal data into numerical feature representation that encodes the linguistic meaning of each word. The second step will be to apply this feature representation to a fully connected multilayer perceptron with an output layer of size 5 corresponding to the numerical (1-5) “rating” that this review would likely be paired with. 

For the first step, we will use a pre-trained Bidirectional Encoder Representations from Transformers (BERT), which is a transformer architecture designed to encode words to numerical features while retaining linguistic context. We use padding in the BERT so that the output of the BERT is a fixed length. We  then add on a multilayer perceptron of our own design. 
