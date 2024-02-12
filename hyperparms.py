'''Configuration file for hyperparameters.'''

import sys
import logging
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)


class Config:

  '''
    Configuration Class, This class holds our hyperparameter values
  '''

  def __init__(self, num_heads, num_layers, emb_dim, seq_length, vocab_size, head_size, pos_embedding, agg_method, pos_embedding_type):

    self.dic = {
    'num_heads' : num_heads,
    'num_att_layers': num_layers,
    'emb_dim': emb_dim,
    'seq_length': seq_length,
    'vocab_size': vocab_size,
    'head_size': head_size, #size of single dense layer head
    'pos_embedding': pos_embedding, #True or False weather to learn positional embeddings
    'agg_method': agg_method, #one of TOKEN or SUM
    'pos_embedding_type':pos_embedding_type, #one of RANDOM or SIN_COS
    'log_dir': "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
    'lr_range': (0.0000000001, 10) #LR range to find a good lr
    }


  def __getitem__(self, name):

    try:
      return(self.dic[name])
    except:
      logger.info('!!!!!!!! {}, NO Such PARAMETER PRESENT !!!!!!!!!!!'.format(name))
