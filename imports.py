import random
from random import sample
from tqdm.auto import tqdm
import pandas as pd
from collections import defaultdict
import numpy as np
import tensorflow as tf
import itertools
import math
import time
from tensorflow.keras.layers import Embedding, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np


config = {
    'num_heads' : 1,
    'num_layers': 1,
    'emb_dim': 128,
    'seq_length': 512,
    'vocab_size': 2,
    'head_size': 64, #size of single dense layer head
    'pos_embedding': True, #True or False weather to learn positional embeddings
    'agg_method': 'TOKEN' #one of TOKEN or SUM
}
