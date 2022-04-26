from tensorflow.keras.layers import Embedding, Dense, Input
import tensorflow as tf
import numpy as np


class Encodings(tf.keras.layers.Layer):

  '''
    This class takes return the embeddings with positional embeddings added.

    Alternating sin cos  positional embeddings are used, the position matrix P is filled using the following rule
    P(k, 2i) = sin(k/n^(2i/d))
    P(k, 2i + 1) = cos(k/n^(2i/d))

    Attributes:
    	emb_dim: the required dimension for our resultant embeddings
    	seq_length: the length of the input sequence
    	vocab_size: the size of our vocab
    	pos_embedding: Boolean flag used to weather include positional encodings or not
    	embedding_type: RANDOM/SIN_COS which type of positional embedding to use , random or alternate sin cos

  '''

  def __init__(self, emb_dim, seq_length, vocab_size, pos_embedding, embedding_type):

    '''Initializes the class with variables and Embedding layers '''

    super().__init__()
    self.embedding_dim = emb_dim
    self.seq_length = seq_length
    self.vocab_size = vocab_size
    self.pos_embedding_flag = pos_embedding
    self.embedding_type = embedding_type
    self.batch_size = 64
    self.embedding = Embedding(self.vocab_size + 2, self.embedding_dim, input_length = self.seq_length)
    self.random_pos_embedding = Embedding(self.seq_length + 1, self.embedding_dim)
    self.sin_pos_embedding_mat = self.get_sin_pos_emb_mat(self.seq_length, self.embedding_dim)
    self.sinusodial_pos_embedding = Embedding(self.seq_length + 1, self.embedding_dim, weights = [self.sin_pos_embedding_mat], trainable= False)





  def get_sin_pos_emb_mat(self, seq_len, d, n = 10000):

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

    '''Performs transformations on our given batch of data'''

    embedding_out = self.embedding(batch)

    if(self.pos_embedding_flag and self.embedding_type == 'SIN_COS'):
      pos_embedding = self.sinusodial_pos_embedding(tf.range(self.seq_length + 1))
      embedding_out = embedding_out + pos_embedding
    else:
      pos_embedding = self.random_pos_embedding(tf.range(self.seq_length + 1))
      embedding_out = embedding_out + pos_embedding


    return embedding_out



class Attention(tf.keras.layers.Layer):

  '''
    This class implements the Multi head Attention layer mechanism and returns the attention output and attention scores for each attention head

    The query, key, and value matrices are first calculated and then the attention scores are calculated with each input token as a query. The following operations are performed

    	softmax((Q*K_T)/sqrt(attention_head_size))*V

    	Q: Query matrix
    	K_T: Key matrix transpose
    	V: Value matrix

    Attributes:
    	num_heads: Number of heads we want in out attention layer
    	emb_dim: the required dimension for our resultant embeddings
  '''

  def __init__(self, num_heads, emb_dim):

    '''Initiates all the variables and layers required to perform attention'''

    super().__init__()

    self.num_att_heads = num_heads
    self.attention_head_size = int(emb_dim / self.num_att_heads)
    self.all_head_size = self.num_att_heads * self.attention_head_size

    self.query = Dense(self.all_head_size)
    self.key = Dense(self.all_head_size)
    self.value = Dense(self.all_head_size)
    self.out = Dense(emb_dim)

  def split_heads(self, input_layer, hidden_states_shape):

    '''Splits the input matrix between attention heads'''

    return tf.reshape(input_layer, (hidden_states_shape[0], self.num_att_heads,
                                 hidden_states_shape[1],
                                 self.attention_head_size))

  def call(self, hidden_states):

    '''Performs transformations on input batch to get our attention scores and output'''

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

  ''' This is a wrapper class for attention and encoding class

  	Our output from the attention layer is passed through perceptron layer for transformation,
  	output of which is returned.

     Attributes:

     	emb_dim: the required dimension for our resultant embeddings
    	seq_length: the length of the input sequence
    	vocab_size: the size of our vocab
    	pos_embedding: Boolean flag used to weather include positional encodings or not
    	num_heads: Number of heads we want in out attention layer
    	head_size: the number of units we want in our final dense layer
    	agg_method: TOKEN/SUM tells us how to aggregate our attention output to feed to final dense layer.
    	            TOKEN: the CLS token vectors are fetched and fed into the dense layer
    	            SUM: The attention output is added and frd into the dense layer


  '''

  def __init__(self, emb_dim, seq_length, vocab_size, pos_embedding, num_heads, head_size, agg_method, embedding_type ):

    '''This initializes the variables andlayers required'''

    super().__init__()

    self.encodings = Encodings(emb_dim, seq_length, vocab_size, pos_embedding, embedding_type)
    self.attention = Attention(num_heads, emb_dim)
    self.agg_method = agg_method
    self.head_dim = head_size
    self.head = Dense(self.head_dim)


  def call(self, input):

    '''Performs all the transformations on input batch and returns the attention scores and output'''

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
