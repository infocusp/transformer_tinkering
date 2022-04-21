from imports.py import tf
from imports.py import Dense
from encoding_class.py import Encodings
from attention_class.py import Attention



class _Model(tf.keras.layers.Layer):

  def __init__(self, config):
    super().__init__()

    self.encodings = Encodings(config)
    self.attention = Attention(config)
    self.agg_method = config['agg_method']
    self.head_dim = config['head_size']
    self.head = Dense(self.head_dim)


  def call(self, input):
    op = self.encodings(input)
    att_op, att_scores = self.attention(op)

    if(self.agg_method == 'TOKEN'):
      #op = att_op[:,0,:]
      op = tf.squeeze(tf.gather(att_op, [0], axis=1))
    else:
      #op = tf.math.reduce_sum(att_op[:, 1:, :], axis = 1)
      op = tf.reduce_sum(tf.gather(att_op, list(range(1, att_op.shape[1])), axis=1), axis=1)

    op = self.head(op)


    return op, att_scores


