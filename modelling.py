from imports.py import Embedding, Dense, Input
from imports.py import Model, Adam, K
from imports.py import plt, ReduceLROnPlateau, np
from config.py import Config
from data_generator.py import Generate_data
from model_class.py import _Model


input_shape = (512)
config = Config()
input = Input(input_shape)
op, att_scores = _Model(config)(input)
output = Dense(1, activation='linear')(op)

model = Model(inputs = input, outputs=output)

model.summary()