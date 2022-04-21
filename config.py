
class Config:

  '''
    Configuration Class, This class holds our hyperparameter values
  '''

  def __init__(self):

    self.dic = {
    'num_heads' : 1,
    'num_layers': 1,
    'emb_dim': 128,
    'seq_length': 512,
    'vocab_size': 2,
    'head_size': 64, #size of single dense layer head
    'pos_embedding': True, #True or False weather to learn positional embeddings
    'agg_method': 'SUM' #one of TOKEN or SUM
    }
    

  def __getitem__(self, name):

    try:
      return(self.dic[name])
    except:
      print('!!!!!!!! NO Such PARAMETER PRESENT !!!!!!!!!!!')