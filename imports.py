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