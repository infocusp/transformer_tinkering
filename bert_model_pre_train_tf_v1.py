import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tf.get_logger().setLevel('ERROR')
import math
import warnings
warnings.filterwarnings("ignore")

# Encoding layer
class Encodings(layers.Layer):
    def __init__(self, emb_dim, seq_length, vocab_size, pos_embedding, embedding_type):
        super(Encodings, self).__init__()

        self.embedding_dim = emb_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.pos_embedding_flag = pos_embedding
        self.embedding_type = embedding_type

        # Define embedding layers
        self.embedding = Embedding(vocab_size, self.embedding_dim)
        self.segment_embedding = Embedding(2, self.embedding_dim)

    def sinusodial_pos_embedding(self, max_length):
        # Generate sinusoidal position embeddings
        pos = np.arange(max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embedding_dim, 2) * -(np.log(10000.0) / self.embedding_dim))
        pos_emb = pos * div_term
        pos_emb = np.stack([np.sin(pos_emb), np.cos(pos_emb)], axis=1).reshape(max_length, -1)
        pos_emb[1:, 1::2] = 0
        pos_emb = tf.convert_to_tensor(pos_emb, dtype=tf.float32)
        return tf.Variable(initial_value=pos_emb, trainable=False, dtype=tf.float32)

    def random_pos_embedding(self, max_length):
        # Generate random position embeddings
        pos_emb = tf.random.uniform(shape=(max_length, self.embedding_dim))
        return tf.Variable(initial_value=pos_emb, trainable=True, dtype=tf.float32)

    def call(self, batch, segment_ids):
        # Perform the forward pass of the layer
        embedding_out = self.embedding(batch)
        embedding_out *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))

        if self.pos_embedding_flag and self.embedding_type == 'SIN_COS':
            # Use sinusoidal position embeddings
            pos_embedding = self.sinusodial_pos_embedding(self.seq_length + 1)
        else:
            # Use random position embeddings
            pos_embedding = self.random_pos_embedding(self.seq_length + 1)

        pos_embedding = tf.expand_dims(pos_embedding[:tf.shape(batch)[1]], axis=0)
        pos_embedding = tf.tile(pos_embedding, [tf.shape(batch)[0], 1, 1])

        embedding_out = embedding_out + pos_embedding
        segment_embeds = self.segment_embedding(segment_ids)
        embedding_out = embedding_out + segment_embeds

        return embedding_out
    
# Multi-Head Attention Layer

class MultiAttention(layers.Layer):
    def __init__(self, num_heads, emb_dim):
        super(MultiAttention, self).__init__()
        self.num_att_heads = num_heads
        self.attention_head_size = int(emb_dim / self.num_att_heads)
        self.all_head_size = self.num_att_heads * self.attention_head_size
        self.emb_dim = emb_dim
        self.query = layers.Dense(self.all_head_size)  # Query projection layer
        self.key = layers.Dense(self.all_head_size)  # Key projection layer
        self.value = layers.Dense(self.all_head_size)  # Value projection layer
        self.out = layers.Dense(emb_dim)  # Output projection layer

    def split_heads(self, input_layer, hidden_states_shape):
        # Reshape input tensor to separate heads
        return tf.transpose(tf.reshape(input_layer, (hidden_states_shape[0],
                                                     -1,
                                                     self.num_att_heads,
                                                     self.attention_head_size)),
                            perm=[0, 2, 1, 3])

    def call(self, hidden_states):
        # Perform the forward pass of the layer
        mixed_query_layer = self.query(hidden_states)  # Apply query projection
        mixed_key_layer = self.key(hidden_states)  # Apply key projection
        mixed_value_layer = self.value(hidden_states)  # Apply value projection

        hidden_states_shape = tf.shape(hidden_states)

        query_layer = self.split_heads(mixed_query_layer, hidden_states_shape)  # Split heads for queries
        key_layer = self.split_heads(mixed_key_layer, hidden_states_shape)  # Split heads for keys
        value_layer = self.split_heads(mixed_value_layer, hidden_states_shape)  # Split heads for values

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)  # Compute attention scores
        attention_scores = attention_scores / tf.math.sqrt(float(self.attention_head_size))  # Scale by square root of the head size

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)  # Apply softmax to get attention probabilities

        context_layer = tf.transpose(tf.matmul(attention_probs, value_layer), perm=[0, 2, 1, 3])  # Apply attention to values

        context_layer = tf.reshape(context_layer, shape=(hidden_states_shape[0],
                                                         -1,
                                                         self.emb_dim))  # Reshape back to original shape

        att_output = self.out(context_layer)  # Apply output projection layer

        return att_output, attention_probs
class FeedForwardNetwork(layers.Layer):
    def __init__(self, hidden_size, intermediate_size):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate = layers.Dense(intermediate_size, activation='relu')  # Intermediate dense layer
        self.output_layer = layers.Dense(hidden_size)  # Output dense layer
        self.layer_norm = layers.LayerNormalization()  # Layer normalization

    def call(self, inputs):
        hidden = self.intermediate(inputs)  # Apply intermediate dense layer
        outputs = self.output_layer(hidden)  # Apply output dense layer
        outputs = self.layer_norm(outputs + inputs)  # Add residual connection and apply layer normalization
        return outputs


class MaskedLanguageModel(layers.Layer):
    def __init__(self, hidden_size, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.dense = layers.Dense(hidden_size)  # Dense layer
        self.activation = tf.keras.activations.gelu  # Activation function (GELU)
        self.layer_norm = layers.LayerNormalization()  # Layer normalization
        self.mlm_output = layers.Dense(vocab_size)  # Output dense layer for masked language modeling

    def call(self, inputs):
        hidden_states = self.dense(inputs)  # Apply dense layer
        hidden_states = self.activation(hidden_states)  # Apply activation function
        hidden_states = self.layer_norm(hidden_states)  # Apply layer normalization
        logits = self.mlm_output(hidden_states)  # Generate logits for masked language modeling

        return logits


class NextSentencePrediction(layers.Layer):
    def __init__(self, hidden_size):
        super(NextSentencePrediction, self).__init__()
        self.hidden_size = hidden_size
        self.classifier = layers.Dense(2)  # Dense layer for next sentence prediction
        self.activation = layers.Softmax()  # Activation function (softmax)

    def call(self, inputs):
        # Add a check for input dimensions
        if len(inputs.shape) == 2:
            logits = self.classifier(inputs)  # Apply dense layer to inputs
        else:
            logits = self.classifier(inputs[:, 0, :])  # Apply dense layer to first token in inputs
        logits = self.activation(logits)  # Apply activation function (softmax) to get probabilities
        return logits
class TransformerModel(tf.keras.layers.Layer):
    def __init__(self, emb_dim, seq_length, vocab_size, pos_embedding, num_heads, head_size, agg_method, embedding_type,
                 num_att_layers, intermediate_size):
        super(TransformerModel, self).__init__()

        self.encodings = Encodings(emb_dim, seq_length, vocab_size, pos_embedding, embedding_type)
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.agg_method = agg_method
        self.head_dim = head_size
        self.num_att_layers = num_att_layers
        self.feed_forward = FeedForwardNetwork(self.emb_dim, intermediate_size)
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()

    def build(self, input_shape):
        self.att_layers = []
        for _ in range(self.num_att_layers):
            self.att_layers.append(MultiAttention(self.num_heads, self.emb_dim))

    def call(self, inputs):
        input_data, segment_ids = inputs
        op = self.encodings(input_data, segment_ids)  # Obtain the embeddings from the input data

        att_op = op
        att_scores_list = []
        for i in range(self.num_att_layers):
            att_op, att_scores = self.att_layers[i](att_op)  # Apply multi-head attention
            att_scores_list.append(att_scores)  # Store the attention scores

        attention_output = self.layer_norm1(att_op + op)  # Add residual connection and apply layer normalization

        feed_forward_output = self.feed_forward(attention_output)  # Apply feed-forward network
        output = self.layer_norm2(feed_forward_output + attention_output)  # Add residual connection and apply layer normalization

        return output, tf.convert_to_tensor(att_scores_list, dtype=tf.float32)


class BERTModel(tf.keras.Model):
    def __init__(self, emb_dim, seq_length, vocab_size, pos_embedding, num_heads, head_size, agg_method, embedding_type,
                 num_att_layers, intermediate_size):
        super(BERTModel, self).__init__()
        self.transformer = TransformerModel(emb_dim, seq_length, vocab_size, pos_embedding, num_heads, head_size,
                                            agg_method, embedding_type, num_att_layers, intermediate_size)
        self.masked_lm = MaskedLanguageModel(emb_dim, vocab_size)  # Masked Language Model for predicting masked tokens
        self.nsp = NextSentencePrediction(emb_dim)  # Next Sentence Prediction for predicting if two sentences are consecutive

    def call(self, inputs):
        input_data, segment_ids = inputs
        transformer_output, attention_scores = self.transformer(inputs)  # Obtain the transformer output and attention scores
        mlm_output = self.masked_lm(transformer_output)  # Predict masked tokens using the transformer output
        nsp_output = self.nsp(transformer_output[:, 0, :])  # Predict next sentence using the first token's representation
        return mlm_output, nsp_output, attention_scores

# Sample input data (paragraph)
paragraph = """

In the quiet town of Willowbrook, nestled among rolling hills and blooming meadows, life unfolded at a leisurely pace.
 The scent of freshly baked bread wafted through the air, enticing passersby to step into the cozy bakery on Main Street. 
 Children laughed and played in the park, their carefree spirits dancing with the sunlight filtering through the ancient trees. 
 Elderly couples strolled hand in hand along the winding paths, their love a testament to enduring companionship. 
 As the day melted into dusk, the sky painted itself in hues of orange and pink, casting a warm glow over the town's charming houses and inviting cafes. 
In this idyllic haven, time seemed to stand still, embracing a simplicity and serenity that whispered tales of contentment and peace.

"""

# Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([paragraph])
sequences = tokenizer.texts_to_sequences([paragraph])
input_data = pad_sequences(sequences, maxlen=seq_length, padding='post')
segment_ids = np.zeros_like(input_data)

# Initialize the BERT model
vocab_size = len(tokenizer.word_index) + 1
emb_dim = 768
num_heads = 12
head_size = 64
agg_method = 'mean'
embedding_type = 'SIN_COS'
num_att_layers = 12
intermediate_size = 3072

bert_model = BERTModel(emb_dim, seq_length, vocab_size, True, num_heads, head_size, agg_method, embedding_type,
                       num_att_layers, intermediate_size)

# Define loss function and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Training loop
epochs = 10
batch_size = 1  
total_steps = 1  

for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch+1}/{epochs}")
    epoch_loss = 0.0

    with tqdm(total=total_steps, desc="Training") as pbar:
        for step in range(total_steps):
            with tf.GradientTape() as tape:
                # Forward pass
                mlm_output, nsp_output, _ = bert_model([input_data, segment_ids])

                # Compute loss
                mlm_labels = input_data
                nsp_labels = np.random.randint(0, 2, size=(batch_size,))
                mlm_loss = loss_object(mlm_labels, mlm_output)
                nsp_loss = loss_object(nsp_labels, nsp_output)
                total_loss = mlm_loss + nsp_loss

            # Backward pass
            gradients = tape.gradient(total_loss, bert_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, bert_model.trainable_variables))

            epoch_loss += total_loss
            pbar.set_postfix({"Loss": total_loss.numpy()})
            pbar.update()

    average_loss = epoch_loss / total_steps
    print(f"Average loss: {average_loss}\n")
