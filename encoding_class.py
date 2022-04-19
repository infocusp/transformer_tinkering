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
    batch = batch + 2
    batch = tf.concat([tf.expand_dims(tf.convert_to_tensor(np.ones((self.batch_size)), dtype='int64'),axis=1), batch], axis=1)
    #embedding_out = tf.concat([self.embedding(batch), tf.expand_dims(self.class_tokens, axis=1)], axis=1)
    embedding_out = self.embedding(batch)

    if(self.pos_embedding_flag):
      pos_embedding = self.pos_embedding(tf.range(start = 0, limit = self.seq_length + 1, delta=1))
      embedding_out = embedding_out + pos_embedding


    return embedding_out
