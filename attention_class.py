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

  def call(self, hidden_states):

    #getting the query , key and value vectors
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    hidden_states_shape = tf.shape(hidden_states)

    #Dividing query keay and value vectors between given number of attention heads
    query_layer = tf.reshape(mixed_query_layer, shape = (hidden_states_shape[0], self.num_att_heads, 
                                 hidden_states_shape[1],
                                 self.attention_head_size))
    

    key_layer = tf.reshape(mixed_key_layer, shape = (hidden_states_shape[0], self.num_att_heads, 
                                 hidden_states_shape[1],
                                 self.attention_head_size))
    
    value_layer = tf.reshape(mixed_value_layer, shape = (hidden_states_shape[0], self.num_att_heads, 
                                 hidden_states_shape[1],
                                 self.attention_head_size))
    
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


