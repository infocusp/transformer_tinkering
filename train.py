from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import  ReduceLROnPlateau
import numpy as np
import hyperparams
import dataset
import model
from dataset.py import Generate_data
from model.py import _Model
import argparse





# input_shape = (512)
# config = hyperparams.Config()
# input = Input(input_shape)
# op, att_scores = model._Model(config)(input)
# output = Dense(1, activation='linear')(op)

# _model = Model(inputs = input, outputs=output)

# _model.summary()


def _train(args):
    input_shape = (args.seq_length)
    config = hyperparams.Config(args.num_heads, args.num_layers, args.emb_dim, args.seq_length, args.vocab_size, args.head_size, args.pos_embedding, args.agg_method, args.pos_embedding_type)
    input = Input(input_shape)
    op, att_scores = model._Model(config['emb_dim'], config['seq_length'], config['vocab_size'], config['pos_embedding'], config['num_heads'], config['head_size'], config['agg_method'],
    config['pos_embedding_type'])(input)
    output = Dense(1, activation='linear')(op)

    _model = Model(inputs = input, outputs=output)
    print(_model.summary())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_heads', type=int, default=1, help='Number of Attention Heads')
    parser.add_arguments('--num_layers', type=int, default=1, help='Number of Attention Layers')
    parser.add_arguments('--emb_dim', type=int, default=128, help='Embedding Dimensions')
    parser.add_arguments('--seq_length', type=int, default=512, help='input seq length')
    parser.add_arguments('--vocab_size', type=int, default=2, help='length of vocab for input')
    parser.add_arguments('--head_size', type=int, default=64, help='Size of the single dense layer head')
    parser.add_arguments('--pos_embedding', type=bool, default=True, help='Weather to use positional embedding or not')
    parser.add_arguments('--agg_method', type=str, default='TOKEN', help='Method used to feed into head. TOKEN Uses CLS token, SUM adds the output from attention')
    parser.add_arguments('--pos_embedding_type', type=str, default='SIN_COS', help='Which type of positional encoding to use. SIN_COS for alternating sin cos, RANDOM for random pos encodings')
    parser.add_arguments('--problem_num', type=int, default=1, help=''' Integer indicating which problem to solve?
    										Distance between 2 red tokens: 1
										Count number of red tokens: 2
										Find token that appears maximum time: 3
										Compute sequence length: 4
										Palindrome Sequence: 5
										Sorted Sequence: 6
										Sum: 7
										MAx: 8
										Min: 9
    										''')
    args = parser.parse_args()

    _train(args)
