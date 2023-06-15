import torch
import torch.nn as nn
import math
from torch.optim import Adam
from tqdm import tqdm

class BERTInputEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length):
        super(BERTInputEncoder, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.segment_embedding = nn.Embedding(2, embedding_dim)

        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

    def forward(self, input_ids, segment_ids):
        # Token embedding
        token_embeds = self.token_embedding(input_ids)
        # Positional encoding
        position_ids = torch.arange(input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        position_embeds = self.position_embedding(position_ids)
        # Segment embedding
        segment_embeds = self.segment_embedding(segment_ids)
        # Summing token, positional, and segment embeddings
        embeddings = token_embeds + position_embeds + segment_embeds

        return embeddings


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)

        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, mask=None):
        batch_size, seq_length, hidden_size = inputs.size()

        # Projecting inputs for query, key, and value
        queries = self.query_projection(inputs).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        keys = self.key_projection(inputs).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        values = self.value_projection(inputs).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size).float())

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Apply attention scores to values
        context = torch.matmul(attention_probs, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)

        # Project back to the original hidden size and apply layer normalization
        outputs = self.output_projection(context)
        outputs = self.layer_norm(outputs)

        return outputs

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(FeedForwardNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        hidden = self.intermediate(inputs)
        hidden = torch.relu(hidden)
        outputs = self.output(hidden)
        outputs = self.layer_norm(outputs)

        return outputs

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super(TransformerEncoderLayer, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size

        self.attention = MultiHeadSelfAttention(hidden_size, num_heads)
        self.feed_forward = FeedForwardNetwork(hidden_size, intermediate_size)

        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.feed_forward_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, mask=None):
        attention_outputs = self.attention_layer_norm(inputs + self.attention(inputs, mask))
        feed_forward_outputs = self.feed_forward_layer_norm(attention_outputs + self.feed_forward(attention_outputs))

        return feed_forward_outputs

class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(MaskedLanguageModel, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        hidden_states = self.dense(inputs)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)

        return logits

class NextSentencePrediction(nn.Module):
    def __init__(self, hidden_size):
        super(NextSentencePrediction, self).__init__()

        self.hidden_size = hidden_size

        self.classifier = nn.Linear(hidden_size, 2)
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        logits = self.classifier(inputs[:, 0, :])
        logits = self.activation(logits)
        return logits


class BERT(nn.Module):
    def __init__(self, vocab_size, max_seq_length, hidden_size, num_heads, intermediate_size, num_layers):
        super(BERT, self).__init__()

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers

        self.embedding = BERTInputEncoder(vocab_size, hidden_size, max_seq_length)

        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)
        ])

        self.mlm = MaskedLanguageModel(hidden_size, vocab_size)
        self.nsp = NextSentencePrediction(hidden_size)

    def forward(self, input_ids, segment_ids, attention_mask):
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

# Initialize the model
model = BERT(VOCAB_SIZE, MAX_SEQ_LENGTH, HIDDEN_SIZE, NUM_HEADS, INTERMEDIATE_SIZE, NUM_LAYERS)

# Determine the device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Move the model to the selected device
model = model.to(device)

# Define a dataset
num_sequences = 100
sequence_length = 20  
input_ids = torch.randint(0, VOCAB_SIZE, size=(num_sequences, sequence_length))
segment_ids = torch.randint(0, 2, size=(num_sequences, sequence_length))
attention_mask = torch.ones((num_sequences, sequence_length))
# attention_mask = torch.ones((num_sequences, sequence_length), dtype=torch.bool)  

# Define target for MLM and NSP tasks
masked_lm_labels = input_ids.clone()
next_sentence_labels = torch.randint(0, 2, size=(num_sequences,))

# Move all data and targets to the device
input_ids = input_ids.to(device)
segment_ids = segment_ids.to(device)
attention_mask = attention_mask.to(device)
masked_lm_labels = masked_lm_labels.to(device)
next_sentence_labels = next_sentence_labels.to(device)

# Loss functions and optimizer
mlm_loss_function = torch.nn.CrossEntropyLoss()
nsp_loss_function = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in tqdm(range(10)):  
    model.train()

    # Forward pass
    mlm_output, nsp_output = model(input_ids, segment_ids, attention_mask)

    # Calculate loss
    mlm_loss = mlm_loss_function(mlm_output.view(-1, VOCAB_SIZE), masked_lm_labels.view(-1))
    nsp_loss = nsp_loss_function(nsp_output, next_sentence_labels)
    total_loss = mlm_loss + nsp_loss

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f'Epoch: {epoch}, Loss: {total_loss.item()}')
