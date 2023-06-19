import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class BERTInputEncoder(layers.Layer):
    def __init__(self, vocab_size, embedding_dim, max_seq_length):
        super(BERTInputEncoder, self).__init__()

        self.token_embedding = layers.Embedding(vocab_size, embedding_dim)
        self.position_embedding = layers.Embedding(max_seq_length, embedding_dim)
        self.segment_embedding = layers.Embedding(2, embedding_dim)

        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

    def call(self, input_ids, segment_ids):
        # Token embedding
        token_embeds = self.token_embedding(input_ids)
        # Positional encoding
        position_ids = tf.range(input_ids.shape[1])
        position_embeds = self.position_embedding(position_ids)
        # Segment embedding
        segment_embeds = self.segment_embedding(segment_ids)
        # Summing token, positional, and segment embeddings
        embeddings = token_embeds + position_embeds + segment_embeds

        return embeddings


class Multi_Attention(layers.Layer):
    def __init__(self, num_heads, emb_dim):
        super().__init__()
        self.num_att_heads = num_heads
        self.attention_head_size = int(emb_dim / self.num_att_heads)
        self.all_head_size = self.num_att_heads * self.attention_head_size
        self.emb_dim = emb_dim
        self.query = layers.Dense(self.all_head_size)  # Dense layer for query projection
        self.key = layers.Dense(self.all_head_size)  # Dense layer for key projection
        self.value = layers.Dense(self.all_head_size)  # Dense layer for value projection
        self.out = layers.Dense(emb_dim)  # Dense layer for output projection

    def split_heads(self, input_layer, hidden_states_shape):
        # Reshape and transpose the input layer to split it into multiple heads
        return tf.transpose(tf.reshape(input_layer, (hidden_states_shape[0],
                                                     -1,
                                                     self.num_att_heads,
                                                     self.attention_head_size)),
                            perm=[0, 2, 1, 3])

    def call(self, hidden_states):
        # Project the input hidden states into query, key, and value representations
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        hidden_states_shape = tf.shape(hidden_states)
        
        # Split the projected layers into multiple heads
        query_layer = self.split_heads(mixed_query_layer, hidden_states_shape)
        key_layer = self.split_heads(mixed_key_layer, hidden_states_shape)
        value_layer = self.split_heads(mixed_value_layer, hidden_states_shape)
        
        # Compute attention scores
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(float(self.attention_head_size))
        
        # Apply softmax to obtain attention probabilities
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        
        # Perform the attention operation by multiplying attention probabilities with value layer
        context_layer = tf.transpose(tf.matmul(attention_probs, value_layer), perm=[0, 2, 1, 3])
        
        # Reshape the context layer to the original shape
        context_layer = tf.reshape(context_layer, shape=(hidden_states_shape[0],
                                                         -1,
                                                         self.emb_dim))
        
        # Project the context layer to the output size
        att_output = self.out(context_layer)
        
        return att_output, attention_probs


class FeedForwardNetwork(layers.Layer):
    def __init__(self, hidden_size, intermediate_size):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
# Intermediate dense layer with ReLU activation
        self.intermediate = layers.Dense(intermediate_size, activation='relu')  
        self.output_layer = layers.Dense(hidden_size)  # Output dense layer
        self.layer_norm = layers.LayerNormalization()  # Layer normalization

    def call(self, inputs):
       # Pass the inputs through the intermediate dense layer
        hidden = self.intermediate(inputs) 
         # Obtain the final output by passing the intermediate output through the output dense layer
        outputs = self.output_layer(hidden) 
         # Apply layer normalization to the sum of the outputs and the inputs
        outputs = self.layer_norm(outputs + inputs) 
        return outputs

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super(TransformerEncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size

        self.attention = Multi_Attention(num_heads, hidden_size)  # Multi-Head Attention layer
        self.feed_forward = FeedForwardNetwork(hidden_size, intermediate_size)  # Feed-Forward Network

        self.layer_norm1 = layers.LayerNormalization()  # Layer normalization for the first sub-layer
        self.layer_norm2 = layers.LayerNormalization()  # Layer normalization for the second sub-layer

    def call(self, inputs, mask=None):
      # Obtain the output from the Multi-Head Attention layer
        attention_output, _ = self.attention(inputs)  
         # Add and normalize the attention output with the input
        attention_output = self.layer_norm1(attention_output + inputs) 
         # Pass the attention output through the Feed-Forward Network
        feed_forward_output = self.feed_forward(attention_output) 
        # Add and normalize the feed-forward output with the attention output
        output = self.layer_norm2(feed_forward_output + attention_output)  

        return output

class MaskedLanguageModel(layers.Layer):
    def __init__(self, hidden_size, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.dense = layers.Dense(hidden_size)  # Fully connected layer
        self.activation = tf.keras.activations.gelu  # GELU activation function
        self.layer_norm = layers.LayerNormalization()  # Layer normalization
        self.mlm_output = layers.Dense(vocab_size)  # Output layer for masked language modeling

    def call(self, inputs):
       # Pass the inputs through the fully connected layer
        hidden_states = self.dense(inputs) 
         # Apply the GELU activation function
        hidden_states = self.activation(hidden_states) 
         # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states) 
         # Generate logits for masked language modeling
        logits = self.mlm_output(hidden_states) 

        return logits

class NextSentencePrediction(layers.Layer):
    def __init__(self, hidden_size):
        super(NextSentencePrediction, self).__init__()
        self.hidden_size = hidden_size
# Fully connected layer for binary classification
        self.classifier = layers.Dense(2)  
         # Softmax activation for probability distribution
        self.activation = layers.Softmax() 

    def call(self, inputs):
      # Extract the first token's representation from inputs
        logits = self.classifier(inputs[:, 0, :])  
        # Apply softmax activation to obtain probabilities
        logits = self.activation(logits)  
        return logits



class BERT(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_length, hidden_size, num_heads, intermediate_size, num_layers):
        super(BERT, self).__init__()

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers

        self.embedding = BERTInputEncoder(vocab_size, hidden_size, max_seq_length)

        self.encoder = [
            TransformerEncoderLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)
        ]

        self.mlm = MaskedLanguageModel(hidden_size, vocab_size)
        self.nsp = NextSentencePrediction(hidden_size)

    def call(self, input_ids, segment_ids, attention_mask):
        # Input Embedding
        embeddings = self.embedding(input_ids, segment_ids)
        # Pass through each Encoder Layer
        for layer in self.encoder:
            embeddings = layer(embeddings, attention_mask)
        # MLM & NSP
        mlm_output = self.mlm(embeddings)
        nsp_output = self.nsp(embeddings)

        return mlm_output, nsp_output



# Parameters
VOCAB_SIZE = 30522
MAX_SEQ_LENGTH = 512
HIDDEN_SIZE = 768
NUM_HEADS = 12
INTERMEDIATE_SIZE = 3072
NUM_LAYERS = 12
learning_rate = 0.01

# Set default batch size and sequence length
BATCH_SIZE = 4
SEQ_LENGTH = 20

# Initialize default inputs
input_ids = np.random.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LENGTH))
segment_ids = np.zeros(shape=(BATCH_SIZE, SEQ_LENGTH))  # Assume single-sequence input 
attention_mask = np.ones(shape=(BATCH_SIZE, SEQ_LENGTH))  
masked_lm_labels = np.random.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LENGTH))
next_sentence_labels = np.zeros(shape=(BATCH_SIZE, 1))  # Assume all sequences are related

# Convert numpy arrays to TensorFlow tensors
input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
segment_ids = tf.convert_to_tensor(segment_ids, dtype=tf.int32)
attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int32)
masked_lm_labels = tf.convert_to_tensor(masked_lm_labels, dtype=tf.int32)
next_sentence_labels = tf.convert_to_tensor(next_sentence_labels, dtype=tf.int32)


# Initialize the model
model = BERT(VOCAB_SIZE, MAX_SEQ_LENGTH, HIDDEN_SIZE, NUM_HEADS, INTERMEDIATE_SIZE, NUM_LAYERS)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Define loss functions
mlm_loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
nsp_loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Training loop
for epoch in range(10):
    with tf.GradientTape() as tape:
        # Forward pass
        mlm_output, nsp_output = model(input_ids, segment_ids, attention_mask)

        # Calculate loss
        mlm_loss = mlm_loss_function(masked_lm_labels, mlm_output)
        nsp_loss = nsp_loss_function(next_sentence_labels, nsp_output)
        total_loss = mlm_loss + nsp_loss

    # Backward pass and optimization
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(f'Epoch: {epoch}, Loss: {total_loss.numpy()}')