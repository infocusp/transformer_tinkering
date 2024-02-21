# Transformer Tinkering


## Description
In these experiments we will test the learning capability of a transformer attention mechanism on some basic problems. We use the usual transformer architecture (encoder only) with a small tweak where we remove the feed forward neural network from the encoder blocks. Removing non-linearity from the blocks will tell us the learning capability of the attention mechanism alone, which we will verify by looking at the attention plots.
We will test the model on some basic problems briefly described below:
1. Find the distance between two red tokens in a variable length sequence of red and black tokens.
2. Test whether we can count the number of red tokens in a variable length sequence of red and black tokens.
3. Find the token that occurs the maximum number of times.
4. Given a variable length sequence, find its length.
5. Test whether the given variable length sequence is a palindrome or not.
6. Test whether a given variable length sequence is sorted or not.
7. Test whether we can compute the sum of the tokens in a sequence.
8. Test that we can compute the maximum of the tokens in a sequence.
9. Test whether we can compute the minimum of the tokens in a sequence.


## Details
For a particular experiment we first generate the data which is used for training and evaluation purposes. Dataset generation utilities can be found in the dataset module. Once the data is generated we train our model for a specific problem, while training we save the attention plots for each epoch. Looking at the attention plots we actually get some idea about the things the model is attending to.

Below plot shows the model attending to important tokens required to solve problem 3.
| ![](https://github.com/InFoCusp/transformer_tinkering/blob/main/problem_3_infer.png?raw=true) |
|:--:| 
| *Attention plots for each head showing model attending to the token occuring maximum number of times (24 in this case).* |

Another one below shows the attention plot where the model is attending to important token required to solve the problem 8, which is maximum number from the sequence.
| ![](https://github.com/InFoCusp/transformer_tinkering/blob/main/problem_8_infer(2).png?raw=true) |
|:--:| 
| *Attention plots showing model heads attending to the maximum token 9 in this case. (Note 10 is used for CLS token)* |

On similar lines other problems were solved and a detailed blog about it can be found [here](https://medium.com/@divyam.vashisht/learning-capability-of-a-transformer-43261a9dc77a). There are other details e.g training loss curves , training time plots, model architecture details which can be found in the blog

##


