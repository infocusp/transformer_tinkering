import sys
import logging

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)



class Config:

  '''
    Configuration Class, This class holds our hyperparameter values
  '''

  def __init__(self, num_heads, num_layers, emb_dim, seq_length, vocab_size, head_size, pos_embedding, agg_method, pos_embedding_type):

    self.dic = {
    'num_heads' : num_heads,
    'num_layers': num_layers,
    'emb_dim': emb_dim,
    'seq_length': seq_length,
    'vocab_size': vocab_size,
    'head_size': head_size, #size of single dense layer head
    'pos_embedding': pos_embedding, #True or False weather to learn positional embeddings
    'agg_method': agg_method, #one of TOKEN or SUM
    'pos_embedding_type':SIN_COS #one of RANDOM or SIN_COS
    }


  def __getitem__(self, name):

    try:
      return(self.dic[name])
    except:
      logger.info('!!!!!!!! NO Such PARAMETER PRESENT !!!!!!!!!!!')
