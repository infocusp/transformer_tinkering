from imports.py import *
from data_generator.py import *
from encoding_class.py import *
from attention_class.py import *
from model_class.py import *


input_shape = (512)
input = Input(input_shape)
op, att_scores = _Model(config)(input)
output = Dense(1, activation='linear')(op)

model = Model(inputs = input, outputs=output)

model.summary()