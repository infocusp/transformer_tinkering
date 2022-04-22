from hyperparms.py import Embedding, Dense, Input
from hyperparams.py import Model, Adam, K
from hyperparams.py import plt, ReduceLROnPlateau, np
from hyperparams.py import Config
from dataset.py import Generate_data
from model.py import _Model


input_shape = (512)
config = Config()
input = Input(input_shape)
op, att_scores = _Model(config)(input)
output = Dense(1, activation='linear')(op)

model = Model(inputs = input, outputs=output)

model.summary()
