# Transformer Tinkering


## Description
In these experiments we will test the learning capability of a transformer attention mechanism on some basic problems.
We use the usual transformer architecture (encoder only) with a small tweak where we remove the feed forward neural network from the encoder blocks. Removing non-linearity from the blocks will tell us the learning capability of the attention mechanism alone, which we will verify by looking at the attention plots.

We will test the model on some basic problems briefly described below:
1. Find the distance between two red tokens in a variable length sequence of red and black tokens.
2. Test weather we can find count the number of red tokens in a variable length sequence of red and black tokens.
3. Find the token that occurs maximum number of times.
4. Given a variable length sequence find its length.
5. Test weather given variable length sequence is a palindrome or not.
6. Test weather given variable length sequence is a sorted or not.
7. Test weather we can compute the sum of the tokens in a sequence.
8. Test that we can compute maximum of the tokens in a sequence.
9. Test weather we can compute minimum of the tokens in a sequence.

## Details
For a particular experiment we first generate the data which is used for training and evaluation purposes. Dataset generation utilities can be found in dataset module. Once the data is generated we train our model for a specific problem, while training we save the attention plots for each epoch. Looking at the attention plots we actually get some idea about the things the model is attending to.

Below plot shows the model attending to important tokens required to get distance between them for problem 1 in a two head setting.
![Problem 1](https://github.com/InFoCusp/transformer_tinkering/blob/main/problem_1_infer(1).png?raw=true)

Another one below shows the attention plot where the model is attending to important token required to solve the problem 8, which is maximum number from the sequence.
![Problem 8](https://github.com/InFoCusp/transformer_tinkering/blob/main/problem_8_infer(1).png?raw=true)

On similar lines other problems were solved and a detailed blog about it can be found [here](https://medium.com/@divyam.vashisht/learning-capability-of-a-transformer-43261a9dc77a). There are other details e.g training loss curves , training time plots, model architecture details which can be found in the blog

##


