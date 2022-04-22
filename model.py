from hyperparms.py import tf
from hyperparms.py import Dense
from hyperparms.py import Embedding
from hyperparms.py import np



class Encodings(tf.keras.layers.Layer):

  '''
    This class takes input tokens and return the embeddings with class token appended and positional embeddings added.

    Alternating sin cos  positional embeddings are used, the position matrix P is filled using the following rule
    P(k, 2i) = sin(k/n^(2i/d))
    P(k, 2i + 1) = cos(k/n^(2i/d))

  '''

  def __init__(self, config):

    super().__init__()
    self.embedding_dim = config['emb_dim']
    self.seq_length = config['seq_length']
    self.vocab_size = config['vocab_size']
    self.pos_embedding_flag = config['pos_embedding']
    self.batch_size = 64
    self.embedding = Embedding(self.vocab_size + 2, self.embedding_dim, input_length = self.seq_length)
    #self.pos_embedding = Embedding(self.seq_length + 1, self.embedding_dim)
    self.pos_embedding_mat = self.get_pos_emb_mat(self.seq_length, self.embedding_dim)
    self.pos_embedding = Embedding(self.seq_length + 1, self.embedding_dim, weights = [self.pos_embedding_mat], trainable= False)
    

  def get_pos_emb_mat(self, seq_len, d, n = 10000):

    '''
      This function gives the positional embedding matrix 
    '''

    P = np.zeros((seq_len + 1, d))
    for k in range(seq_len + 1):
      for i in range(d // 2):
        denominator = np.power(n, 2*i/d)
        P[k,2*i] = np.sin(k/denominator)
        P[k, 2*i + 1] = np.cos(k/denominator)
    return P

  # def build(self, input_shape):
  #   batch_size = 64
  #   self.class_tokens = Embedding(batch_size, self.embedding_dim)
  #   self.class_tokens = self.class_tokens(tf.range(start=0, limit=batch_size, delta=1))


  def call(self, batch):
    embedding_out = self.embedding(batch)

    if(self.pos_embedding_flag):
      pos_embedding = self.pos_embedding(tf.range(start = 0, limit = self.seq_length + 1, delta=1))
      embedding_out = embedding_out + pos_embedding


    return embedding_out



class Attention(tf.keras.layers.Layer):

  '''
    This class implements the Attention layer mechanism and returns the attention output and attention scores for each attention head
  '''

  def __init__(self,config):
    super().__init__()

    self.num_att_heads = config['num_heads']
    self.attention_head_size = int(config['emb_dim'] / self.num_att_heads)
    self.all_head_size = self.num_att_heads * self.attention_head_size

    self.query = Dense(self.all_head_size)
    self.key = Dense(self.all_head_size)
    self.value = Dense(self.all_head_size)
    self.out = Dense(config['emb_dim'])

  def split_heads(self, input_layer, hidden_states_shape):

    return tf.reshape(input_layer, (hidden_states_shape[0], self.num_att_heads, 
                                 hidden_states_shape[1],
                                 self.attention_head_size))

  def call(self, hidden_states):

    #getting the query , key and value vectors
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    hidden_states_shape = tf.shape(hidden_states)


    #Dividing query keay and value vectors between given number of attention heads
    # query_layer = tf.reshape(mixed_query_layer, shape = (hidden_states_shape[0], self.num_att_heads, 
    #                              hidden_states_shape[1],
    #                              self.attention_head_size))
    

    # key_layer = tf.reshape(mixed_key_layer, shape = (hidden_states_shape[0], self.num_att_heads, 
    #                              hidden_states_shape[1],
    #                              self.attention_head_size))
    
    # value_layer = tf.reshape(mixed_value_layer, shape = (hidden_states_shape[0], self.num_att_heads, 
    #                              hidden_states_shape[1],
    #                              self.attention_head_size))

    query_layer = self.split_heads(mixed_query_layer, hidden_states_shape)

    key_layer = self.split_heads(mixed_key_layer, hidden_states_shape)
    
    value_layer = self.split_heads(mixed_value_layer, hidden_states_shape)



    #getting the attention scores
    attention_scores = tf.matmul(query_layer, tf.transpose(key_layer, perm=[0,1,3,2]))

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    attention_probs = tf.nn.softmax(attention_scores, axis=-1)

    #getting the attention output
    context_layer = tf.matmul(attention_probs, value_layer)
    context_layer = tf.reshape(context_layer, shape=( hidden_states_shape[0],
                                                         hidden_states_shape[1],
                                                         hidden_states_shape[2]))
    
    att_output = self.out(context_layer)

    return att_output, attention_probs




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
      op = tf.gather(att_op, 0, axis=1)
    else:
      #op = tf.math.reduce_sum(att_op[:, 1:, :], axis = 1)
      op = tf.reduce_sum(att_op, axis=1)

    op = self.head(op)


    return op, att_scores




